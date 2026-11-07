@echo off
setlocal enabledelayedexpansion

for /f "delims=" %%i in ('rustc --print=sysroot') do set rustc_sysroot=%%i

set rust_etc=!rustc_sysroot!\lib\rustlib\etc

set natvis_cmd=.nvload !rust_etc!\intrinsic.natvis; .nvload !rust_etc!\liballoc.natvis; .nvload !rust_etc!\libcore.natvis;

where windbgx >nul 2>&1
if !errorlevel! equ 0 (
    if defined VERBOSE echo Launching windbgx...
    windbgx -c "!natvis_cmd!" %*
    exit /b !errorlevel!
)

where windbg >nul 2>&1
if !errorlevel! equ 0 (
    if defined VERBOSE echo Launching windbg...
    windbg -c "!natvis_cmd!" %*
    exit /b !errorlevel!
)

echo Error: Neither windbgx nor windbg found in PATH
echo Please install Windows Debugger or add it to your PATH
exit /b 1
