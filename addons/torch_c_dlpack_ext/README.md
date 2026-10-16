<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# Torch C DLPack Extension

This folder contains the source for the `torch-c-dlpack-ext` package, which provides an
Ahead-Of-Time (AOT) compiled module to support faster DLPack conversion in DLPack v1.2.
By default, `tvm-ffi` will JIT-compile a version of this functionality during loading,
and use a safe-fallback if JIT-compilation fails.
Installing this wheel allows users to avoid this JIT compilation overhead and
also avoid the cases where the user environment does not necessarily have a compiler toolchain
to run JIT-compilation.

```bash
pip install torch-c-dlpack-ext
```

## How the wheel is built

The AOT build has two stages:

1. `tvm_ffi.utils._build_optional_torch_c_dlpack` compiles the C++ addon against
   the installed PyTorch and Python headers and libraries. On Windows, it uses
   Ninja and MSVC to produce a DLL for each PyTorch major/minor version and backend
   (CPU or CUDA).
2. The DLLs are placed in `torch_c_dlpack_ext`, and the setuptools backend in
   `build_backend.py` packages them into a wheel. The release scripts then set the
   Python, ABI, and platform tags to match the compiled libraries.

`build_aot_wheels.sh` and `build_aot_wheels.bat` automate this for the release
workflow. The Windows batch script builds the existing x64 CPU/CUDA release
matrix, or a native ARM64 CPU wheel using the instructions below.

## Build a Windows ARM64 CPU wheel

Prerequisites: a native Windows ARM64 machine, `uv`, and Visual Studio 2022 with
the MSVC ARM64 C++ tools and a Windows SDK. The script initializes the ARM64
compiler environment and selects native CPython 3.12 (not x64 Python under
emulation). Only standard, non-free-threaded CPython 3.12 is currently supported
by this ARM64 release path.

Run from the repository root in PowerShell. The exact PyTorch version below is
also pinned in the release workflow:

```powershell
git submodule update --init --recursive
$torchVersion = "2.16.0.dev20261016+cu134"
.\addons\torch_c_dlpack_ext\build_aot_wheels.bat ARM64 cp312 $torchVersion
```

The script creates `.venv\torch-c-dlpack-arm64-cp312`, installs PyTorch from
NVIDIA's Windows ARM64 nightly index, and builds TVM FFI and the CPU addon. It
packages in a fresh staging directory so existing x64 or other Python ABI DLLs
cannot leak into the wheel. Build failures stop the script.

Outputs:

- Addon wheel: `addons\torch_c_dlpack_ext\wheelhouse\*-cp312-cp312-win_arm64.whl`.
- TVM FFI wheel for testing: `build\wheel-arm64-cp312\dist\*.whl`.

The addon links PyTorch's C++ ABI: use the same exact PyTorch build when installing
and testing the wheel, and rebuild after changing it. Installing the NVIDIA
nightly explicitly is necessary when it is not available from the default package
index. Do not assume compatibility with other nightlies sharing its major/minor
version.

This path builds **CPU only**, even though the PyTorch wheel includes CUDA
libraries. No CUDA toolkit is required. The addon loader selects a CUDA DLL when
`torch.cuda.is_available()` is true, so this CPU-only wheel is not sufficient for
CUDA-enabled use. CUDA on Windows ARM64 is not covered by this release job.

If PyTorch already provides `torch.Tensor.__dlpack_c_exchange_api__` or
`torch.Tensor.__c_dlpack_exchange_api__`, the addon is unnecessary for normal use.

## Test the Windows ARM64 wheel

Use a separate environment to test the installed wheels, rather than the source
tree or a JIT-cached DLL. Continue from the repository root:

```powershell
uv venv .venv\torch-c-dlpack-test-arm64-cp312 --python cpython-3.12-windows-aarch64-none
$testPython = ".venv\torch-c-dlpack-test-arm64-cp312\Scripts\python.exe"
uv pip install --no-config --python $testPython --prerelease=allow `
    --index https://pypi.nvidia.com/nvtorch_oot_nightly/ `
    "torch==$torchVersion" pytest pytest-xdist numpy setuptools ninja
$core = Get-ChildItem .\build\wheel-arm64-cp312\dist\apache_tvm_ffi-*-cp312-cp312-win_arm64.whl
$addon = Get-ChildItem .\addons\torch_c_dlpack_ext\wheelhouse\*-cp312-cp312-win_arm64.whl
uv pip install --no-config --python $testPython --force-reinstall --no-deps $core.FullName $addon.FullName
uv run --no-project --python $testPython python -m pytest -q -rs -n 0 `
    addons\torch_c_dlpack_ext\tests\test_wheel.py `
    tests\python\test_optional_torch_c_dlpack.py `
    tests\python\test_tensor.py `
    tests\python\test_container_dlpack_conversion.py `
    tests\python\test_dlpack_exchange_api.py
```

The installed-wheel test checks the wheel tags and the DLL's ARM64 PE machine
type, loads the packaged DLL, and exercises its exchange API with contiguous and
non-contiguous tensors. It checks data, shared storage, and lifetime after the
original tensor is released. JIT compilation and Python DLPack fallback are
disabled in that test, and PyTorch's own exchange API cannot mask a broken addon.

The existing comprehensive exchange-API test has a known Windows expected
failure; GPU-only tests skip on a CPU machine. The installed-wheel test above
must pass independently of those skips.

## Release pipeline

`.github\workflows\torch_c_dlpack.yml` is the manually triggered addon release
workflow. Its `branch` input selects the branch or tag to build in the repository
where the workflow runs. It builds Linux, macOS, Windows x64, and Windows ARM64
wheels plus a source distribution.

The ARM64 job runs on `windows-11-arm`, builds the pinned CPython 3.12/PyTorch
combination above, and runs the installed-wheel and DLPack tests before uploading
its artifact. Expand this matrix only after validating additional combinations.

The `publish` input defaults to `false`: a normal dispatch builds, tests, and
uploads artifacts without publishing. Explicitly enable `publish` to publish all
distributions to PyPI after all build jobs pass, subject to the `pypi` environment's
protection rules and trusted-publishing configuration.
