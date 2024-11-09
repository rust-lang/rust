//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

#[target_feature(enable = "soft-float")]
//~^ERROR: cannot be toggled with
pub unsafe fn my_fun() {}
