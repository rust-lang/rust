//@ compile-flags: --target=aarch64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: aarch64
//@ compile-flags: -Ctarget-feature=-neon
// For now this is just a warning.
//@ build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

//~? WARN target feature `neon` must be enabled to ensure that the ABI of the current target can be implemented correctly
