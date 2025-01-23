//! Ensure ABI-incompatible features cannot be enabled via `#[target_feature]`.
// ignore-tidy-linelength
//@ compile-flags: --crate-type=lib
//@ revisions: x86 riscv
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu
//@[x86] needs-llvm-components: x86
//@[riscv] compile-flags: --target=riscv32e-unknown-none-elf
//@[riscv] needs-llvm-components: riscv
#![feature(no_core, lang_items, riscv_target_feature, x87_target_feature)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized {}

#[cfg_attr(x86, target_feature(enable = "soft-float"))] #[cfg_attr(riscv, target_feature(enable = "d"))]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
