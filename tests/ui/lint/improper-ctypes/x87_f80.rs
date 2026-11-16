#![deny(improper_ctypes_definitions)]
//@ add-minicore
//@ check-pass
//@ revisions: linux64 linux32 win64 win32
//@ [linux64] compile-flags: --target=x86_64-unknown-linux-gnu
//@ [linux32] compile-flags: --target=i686-unknown-linux-gnu
//@ [win64] compile-flags: --target=x86_64-pc-windows-gnu
//@ [win32] compile-flags: --target=i686-pc-windows-gnu
//@ needs-llvm-components: x86
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

// The x87_f80 type is implemented as an ADT, so it needs custom logic
// to be accepted by improper_ctypes.

#[cfg(target_arch = "x86")]
use minicore::arch::x86::x87_f80;
#[cfg(target_arch = "x86_64")]
use minicore::arch::x86_64::x87_f80;
use minicore::*;

#[unsafe(no_mangle)]
extern "C" fn identity(x: x87_f80) -> x87_f80 {
    x
}
