@echo off
setlocal

REM Copyright 2014 The Rust Project Developers. See the COPYRIGHT
REM file at the top-level directory of this distribution and at
REM http://rust-lang.org/COPYRIGHT.
REM
REM Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
REM http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
REM <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
REM option. This file may not be copied, modified, or distributed
REM except according to those terms.

for /f "delims=" %%i in ('rustc --print=sysroot') do set rustc_sysroot=%%i

set rust_etc=%rustc_sysroot%\lib\rustlib\etc

windbg -c ".nvload %rust_etc%\intrinsic.natvis; .nvload %rust_etc%\liballoc.natvis; .nvload %rust_etc%\libcore.natvis;" %*
