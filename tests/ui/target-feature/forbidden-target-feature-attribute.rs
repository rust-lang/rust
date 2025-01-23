//! Ensure "forbidden" target features cannot be enabled via `#[target_feature]`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[target_feature(enable = "forced-atomics")]
//~^ERROR: cannot be enabled with
pub unsafe fn my_fun() {}
