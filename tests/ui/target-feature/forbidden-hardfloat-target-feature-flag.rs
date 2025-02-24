//! Ensure ABI-incompatible features cannot be enabled via `-Ctarget-feature`.
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=+soft-float
// For now this is just a warning.
//@ build-pass
//@error-pattern: must be disabled to ensure that the ABI
#![feature(no_core, lang_items, riscv_target_feature)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}
