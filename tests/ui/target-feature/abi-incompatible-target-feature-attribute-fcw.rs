//@ compile-flags: --crate-type=lib
//@ compile-flags: --target=aarch64-unknown-none-softfloat
//@ needs-llvm-components: aarch64
#![feature(no_core, lang_items)]
#![no_core]
#![deny(aarch64_softfloat_neon)]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[target_feature(enable = "neon")]
//~^ERROR: enabling the `neon` target feature on the current target is unsound
//~|WARN: previously accepted
pub unsafe fn my_fun() {}
