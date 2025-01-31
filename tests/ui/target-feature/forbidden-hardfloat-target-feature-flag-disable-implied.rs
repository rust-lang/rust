//! Ensure that if disabling a target feature implies disabling an ABI-required target feature,
//! we complain.
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-sse
// For now this is just a warning.
//@ build-pass
//@error-pattern: must be enabled to ensure that the ABI
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}
