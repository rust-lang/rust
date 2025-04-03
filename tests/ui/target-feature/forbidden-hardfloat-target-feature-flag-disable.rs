//! Ensure ABI-required features cannot be disabled via `-Ctarget-feature`.
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-x87
// For now this is just a warning.
//@ build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

//~? WARN target feature `x87` must be enabled to ensure that the ABI of the current target can be implemented correctly
//~? WARN unstable feature specified for `-Ctarget-feature`: `x87`
