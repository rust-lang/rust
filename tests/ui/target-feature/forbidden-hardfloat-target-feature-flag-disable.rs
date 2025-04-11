//! Ensure ABI-required features cannot be disabled via `-Ctarget-feature`.
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-x87
// For now this is just a warning.
//@ build-pass

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

//~? WARN target feature `x87` must be enabled to ensure that the ABI of the current target can be implemented correctly
//~? WARN unstable feature specified for `-Ctarget-feature`: `x87`
