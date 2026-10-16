# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Installed-wheel checks for the Windows ARM64 CPU release."""

from __future__ import annotations

import ctypes
import gc
import importlib
import importlib.metadata
import os
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _check_tensor_exchange() -> None:
    import torch  # noqa: PLC0415
    import tvm_ffi  # noqa: PLC0415

    echo = tvm_ffi.get_global_func("testing.echo")
    make_array = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    for non_contiguous in (False, True):
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        if non_contiguous:
            source = source.t()
        expected = source.clone()
        tensor = tvm_ffi.from_dlpack(source)
        assert tensor.data_ptr() == source.data_ptr()
        assert tuple(tensor.shape) == tuple(source.shape)
        assert tuple(tensor.strides) == source.stride()
        returned = echo(source)
        assert isinstance(returned, torch.Tensor)
        assert returned.data_ptr() == source.data_ptr()
        array = make_array(source)
        source.add_(1)
        expected.add_(1)
        torch.testing.assert_close(returned, expected)
        del source, tensor, returned
        gc.collect()
        retained = array[0]
        del array
        gc.collect()
        torch.testing.assert_close(retained, expected)


def _check_arm64_wheel() -> None:
    import torch  # noqa: PLC0415
    from packaging.tags import sys_tags  # noqa: PLC0415

    assert sysconfig.get_platform() == "win-arm64"
    assert not torch.cuda.is_available(), "This wheel is for CPU-only environments"
    metadata = importlib.metadata.distribution("torch-c-dlpack-ext")
    wheel_metadata = metadata.read_text("WHEEL")
    assert wheel_metadata is not None
    assert f"Tag: {next(sys_tags())}" in wheel_metadata.splitlines()
    assert "torch_c_dlpack_ext.core" not in sys.modules

    # Force the AOT loader even when PyTorch already supplies its own exchange API.
    api_owner = SimpleNamespace()
    with patch.object(torch, "Tensor", api_owner):
        core = importlib.import_module("torch_c_dlpack_ext.core")
    assert core._lib is not None, "The installed addon DLL must be loaded"
    library = Path(core._lib._name).resolve()
    package_dir = Path(metadata.locate_file("torch_c_dlpack_ext")).resolve()
    assert library.parent == package_dir
    assert list(package_dir.glob("*.dll")) == [library]
    assert library.name.endswith("-cpu.dll")

    image = library.read_bytes()
    assert image[:2] == b"MZ"
    pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
    assert image[pe_offset : pe_offset + 4] == b"PE\0\0"
    assert struct.unpack_from("<H", image, pe_offset + 4)[0] == 0xAA64, "Expected an ARM64 DLL"

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    api = api_owner.__dlpack_c_exchange_api__
    pointer = get_pointer(api, b"dlpack_exchange_api")
    assert pointer and pointer == core._lib.TorchDLPackExchangeAPIPtr()
    assert pointer == api_owner.__c_dlpack_exchange_api__

    with (
        patch.object(torch.Tensor, "__dlpack_c_exchange_api__", api, create=True),
        patch.object(
            torch.Tensor, "__dlpack__", side_effect=AssertionError("Python DLPack fallback used")
        ),
    ):
        _check_tensor_exchange()
    print(f"Loaded ARM64 AOT DLL: {library.name}; torch={torch.__version__}")


def test_installed_arm64_wheel(tmp_path: Path) -> None:
    """Exercise only the installed package, with JIT and Python fallback disabled."""
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve())],
        cwd=tmp_path,
        env={
            **os.environ,
            "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
            "TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API": "0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    print(result.stdout)


if __name__ == "__main__":
    _check_arm64_wheel()
