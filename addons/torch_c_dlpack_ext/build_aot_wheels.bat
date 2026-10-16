@REM Licensed to the Apache Software Foundation (ASF) under one
@REM or more contributor license agreements.  See the NOTICE file
@REM distributed with this work for additional information
@REM regarding copyright ownership.  The ASF licenses this file
@REM to you under the Apache License, Version 2.0 (the
@REM "License"); you may not use this file except in compliance
@REM with the License.  You may obtain a copy of the License at
@REM
@REM   http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing,
@REM software distributed under the License is distributed on an
@REM "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
@REM KIND, either express or implied.  See the License for the
@REM specific language governing permissions and limitations
@REM under the License.
@echo off
setlocal enabledelayedexpansion

set arch=%~1
set python_version=%~2

set tvm_ffi=%cd%
set torch_c_dlpack_ext=%tvm_ffi%\addons\torch_c_dlpack_ext

if /I "%arch%"=="ARM64" goto build_arm64

if not exist "%tvm_ffi%\.venv" mkdir "%tvm_ffi%\.venv"
if not exist "%tvm_ffi%\lib" mkdir "%tvm_ffi%\lib"
for %%P in (2.4 2.5 2.6 2.7 2.8 2.9) do (
    call :build_libs %%P
)

copy %tvm_ffi%\lib\*.dll %torch_c_dlpack_ext%\torch_c_dlpack_ext
uv venv %tvm_ffi%\.venv\build --python %python_version%
call %tvm_ffi%\.venv\build\Scripts\activate
uv pip install build wheel
cd %torch_c_dlpack_ext%
python -m build -w
dir dist
for %%f in (dist\*.whl) do python -m wheel tags "%%f" --python-tag=%python_version% --abi-tag=%python_version% --platform-tag=win_amd64
dir dist
mkdir wheelhouse
copy dist\*-win_amd64.whl wheelhouse
dir wheelhouse
endlocal
exit /b

:build_arm64
    if not "%python_version%"=="cp312" (
        echo Windows ARM64 currently supports cp312 only. >&2
        exit /b 1
    )
    if "%~3"=="" (
        echo Usage: build_aot_wheels.bat ARM64 cp312 EXACT_TORCH_VERSION >&2
        exit /b 1
    )
    set "torch_version=%~3"
    set "UV_NO_CONFIG=1"
    set "TVM_FFI_DISABLE_TORCH_C_DLPACK=1"
    set "build_env=%tvm_ffi%\.venv\torch-c-dlpack-arm64-%python_version%"

    set "vs_install="
    for /f "usebackq delims=" %%V in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath`) do set "vs_install=%%V"
    if not defined vs_install (
        echo Visual Studio with the MSVC ARM64 C++ tools is required. >&2
        exit /b 1
    )
    call "%vs_install%\Common7\Tools\VsDevCmd.bat" -arch=arm64 || exit /b 1
    if not exist "%build_env%\Scripts\python.exe" (
        uv venv "%build_env%" --python cpython-3.12-windows-aarch64-none || exit /b 1
    )
    call "%build_env%\Scripts\activate.bat" || exit /b 1
    python -c "import sys, sysconfig; assert sys.version_info[:2] == (3, 12) and sysconfig.get_platform() == 'win-arm64', 'Expected ARM64 CPython 3.12'" || exit /b 1
    uv pip install setuptools ninja build wheel packaging || exit /b 1
    uv pip install "torch==%torch_version%" --prerelease=allow --index https://pypi.nvidia.com/nvtorch_oot_nightly/ || exit /b 1
    uv build --wheel . --python "%build_env%\Scripts\python.exe" --out-dir "%tvm_ffi%\build\wheel-arm64-%python_version%\dist" --config-settings=build-dir=build\wheel-arm64-%python_version% || exit /b 1
    for %%F in ("%tvm_ffi%\build\wheel-arm64-%python_version%\dist\apache_tvm_ffi-*-cp312-cp312-win_arm64.whl") do (
        uv pip install --force-reinstall "%%F" || exit /b 1
    )

    @REM Package in a fresh staging directory, never alongside cached x64 or other ABI DLLs.
    set "stage=%torch_c_dlpack_ext%\build\arm64-%python_version%-%RANDOM%-%RANDOM%"
    if exist "%stage%" (
        echo Staging directory already exists: %stage% >&2
        exit /b 1
    )
    mkdir "%stage%\torch_c_dlpack_ext" || exit /b 1
    for %%F in (build_backend.py pyproject.toml README.md LICENSE NOTICE) do (
        copy /y "%torch_c_dlpack_ext%\%%F" "%stage%\" >nul || exit /b 1
    )
    copy /y "%torch_c_dlpack_ext%\torch_c_dlpack_ext\*.py" "%stage%\torch_c_dlpack_ext\" >nul || exit /b 1
    python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir "%stage%\torch_c_dlpack_ext" || exit /b 1
    if not exist "%stage%\torch_c_dlpack_ext\*-cpu.dll" (
        echo No CPU addon DLL was built. >&2
        exit /b 1
    )
    python -m build --wheel --no-isolation --outdir "%stage%\dist" "%stage%" || exit /b 1
    for %%F in ("%stage%\dist\*.whl") do (
        python -m wheel tags "%%F" --python-tag=%python_version% --abi-tag=%python_version% --platform-tag=win_arm64 --remove || exit /b 1
    )
    if not exist "%torch_c_dlpack_ext%\wheelhouse" mkdir "%torch_c_dlpack_ext%\wheelhouse" || exit /b 1
    copy /y "%stage%\dist\*-%python_version%-%python_version%-win_arm64.whl" "%torch_c_dlpack_ext%\wheelhouse\" || exit /b 1
    exit /b 0

:build_libs
    set torch_version=%1
    call :check_availability
    if %errorlevel%==0 (
        call :get_torch_url
        uv venv %tvm_ffi%\.venv\torch%torch_version% --python %python_version%
        call %tvm_ffi%\.venv\torch%torch_version%\Scripts\activate
        uv pip install setuptools ninja
        uv pip install torch==%torch_version% --index-url !torch_url!
        uv pip install -v .
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib --build-with-cuda
        call deactivate
        rmdir -s -q %tvm_ffi%\.venv\torch%torch_version%
    ) else (
        echo Skipping build for torch %torch_version% on %arch% with python %python_version% as it is not available.
    )
    exit /b 0


:check_availability
    if %torch_version%==2.4 (
        if %python_version%==cp313 exit /b 1
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.5 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.6 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.7 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.8 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.9 (
        if %python_version%==cp39 exit /b 1
        exit /b 0
    )
    echo Unknown or unsupported torch version: %torch_version% >&2
    exit /b 1

:get_torch_url
    set cuda_version=
    if %torch_version%==2.4 set cuda_version=cu124
    if %torch_version%==2.5 set cuda_version=cu124
    if %torch_version%==2.6 set cuda_version=cu126
    if %torch_version%==2.7 set cuda_version=cu128
    if %torch_version%==2.8 set cuda_version=cu129
    if %torch_version%==2.9 set cuda_version=cu129
    if defined cuda_version (
        set torch_url=https://download.pytorch.org/whl/%cuda_version%
        exit /b 0
    )
    echo Unknown or unsupported torch version: %torch_version% >&2
    exit /b 1
