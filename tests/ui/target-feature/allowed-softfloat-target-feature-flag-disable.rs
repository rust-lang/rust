//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-x87
//@ build-pass
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

//~? WARN unstable feature specified for `-Ctarget-feature`: `x87`
