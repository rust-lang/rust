//@ compile-flags: --target=aarch64-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: aarch64
//@ compile-flags: -Ctarget-feature=-neon
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

//~? WARN target feature `neon` must be enabled to ensure that the ABI of the current target can be implemented correctly
