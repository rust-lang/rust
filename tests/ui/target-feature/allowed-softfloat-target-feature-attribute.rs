//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ build-pass
#![feature(no_core, lang_items, x87_target_feature)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

#[target_feature(enable = "x87")]
pub unsafe fn my_fun() {}
