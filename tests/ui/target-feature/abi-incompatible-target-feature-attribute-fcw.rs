//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=aarch64-unknown-none-softfloat
//@ needs-llvm-components: aarch64
#![feature(no_core, lang_items)]
#![no_core]
#![deny(aarch64_softfloat_neon)]

#[lang = "sized"]
pub trait Sized {}

#[target_feature(enable = "neon")]
//~^ERROR: enabling the `neon` target feature on the current target is unsound
//~|WARN: previously accepted
pub unsafe fn my_fun() {}
