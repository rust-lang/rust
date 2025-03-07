//! Ensure "forbidden" target features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
#![feature(no_core, lang_items, const_trait_impl)]
#![no_core]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
#[const_trait]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
pub trait Sized: MetaSized {}

#[target_feature(enable = "forced-atomics")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
