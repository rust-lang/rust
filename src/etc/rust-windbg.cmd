@echo off
setlocal enabledelayedexpansion

@rem There is no need to start a debugging session with this script anymore. It is only kept for backwards compatibility.
@rem Let's notify users of this fact and point them to the new way of doing things.
echo This script has been deprecated. Natvis data is now embedded in PDBs. WinDbgX can resolve that without this script. You may want to run the debugger directly.

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
