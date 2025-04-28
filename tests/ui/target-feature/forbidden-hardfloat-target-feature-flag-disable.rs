//! Ensure ABI-required features cannot be disabled via `-Ctarget-feature`.
//! Also covers the case of a feature indirectly disabling another via feature implications.
//@ compile-flags: --crate-type=lib
//@ revisions: x86 x86-implied aarch64
//@[x86] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-x87
//@[x86] needs-llvm-components: x86
//@[x86-implied] compile-flags: --target=x86_64-unknown-linux-gnu -Ctarget-feature=-sse
//@[x86-implied] needs-llvm-components: x86
//@[aarch64] compile-flags: --target=aarch64-unknown-linux-gnu -Ctarget-feature=-neon
//@[aarch64] needs-llvm-components: aarch64
// For now this is just a warning.
//@ build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

//~? WARN must be enabled to ensure that the ABI of the current target can be implemented correctly
//[x86]~? WARN unstable feature specified for `-Ctarget-feature`
