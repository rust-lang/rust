//@ compile-flags: --target i686-unknown-linux-gnu -Zregparm=2 -Zreg-struct-return=true -Cpanic=abort
// Auxiliary build problems with aarch64-apple:
// Shared library linking cc seems to convert "-m32" flag into -arch armv4t
// Auxiliary build problems with i686-mingw: linker `cc` not found
//@ only-x86
//@ ignore-windows
//@ ignore-apple
//@ needs-llvm-components: x86
#![crate_type = "rlib"]
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn somefun() {}

pub struct S;
