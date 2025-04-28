//! Ensure ABI-incompatible features cannot be enabled via `-Ctarget-feature`.
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=+soft-float
// For now this is just a warning.
//@ build-pass

#![feature(no_core, lang_items, riscv_target_feature)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

//~? WARN target feature `soft-float` must be disabled to ensure that the ABI of the current target can be implemented correctl
//~? WARN unstable feature specified for `-Ctarget-feature`: `soft-float`
