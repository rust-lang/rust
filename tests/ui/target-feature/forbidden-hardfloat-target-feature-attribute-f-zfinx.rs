//! Ensure ABI-incompatible features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv64gc-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: riscv
#![feature(no_core, lang_items, riscv_target_feature)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized {}

#[target_feature(enable = "zdinx")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
