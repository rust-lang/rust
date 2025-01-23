//@ compile-flags: --target=i686-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-cpu=pentium
// For now this is just a warning.
//@ build-pass

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

//~? WARN target feature `sse2` must be enabled to ensure that the ABI of the current target can be implemented correctly
