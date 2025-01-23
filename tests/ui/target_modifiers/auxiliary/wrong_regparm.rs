//@ compile-flags: --target i686-unknown-linux-gnu -Zregparm=2 -Cpanic=abort
//@ needs-llvm-components: x86
#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn somefun() {}

pub struct S;
