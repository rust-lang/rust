//! Ensure ABI-incompatible features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
#![feature(no_core, lang_items, riscv_target_feature)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

#[target_feature(enable = "d")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
