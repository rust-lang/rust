@echo off
setlocal

for /f "delims=" %%i in ('rustc --print=sysroot') do set rustc_sysroot=%%i

set rust_etc=%rustc_sysroot%\lib\rustlib\etc

windbg -c ".nvload %rust_etc%\intrinsic.natvis; .nvload %rust_etc%\liballoc.natvis; .nvload %rust_etc%\libcore.natvis;" %*
