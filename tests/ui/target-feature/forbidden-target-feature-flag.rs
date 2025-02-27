//! Ensure "forbidden" target features cannot be enabled via `-Ctarget-feature`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
//@ compile-flags: -Ctarget-feature=+forced-atomics
// For now this is just a warning.
//@ build-pass
//@error-pattern: unsound because it changes the ABI
#![feature(no_core, lang_items, const_trait_impl)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
#[const_trait]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
pub trait Sized: MetaSized {}
