//! Ensure "forbidden" target features cannot be disabled via `-Ctarget-feature`.
//@ compile-flags: --target=riscv32e-unknown-none-elf --crate-type=lib
//@ needs-llvm-components: riscv
//@ compile-flags: -Ctarget-feature=-forced-atomics
// For now this is just a warning.
//@ build-pass
//@error-pattern: unsound because it changes the ABI
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
