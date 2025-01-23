//@ compile-flags: --target=i686-unknown-linux-gnu --crate-type=lib
//@ needs-llvm-components: x86
//@ compile-flags: -Ctarget-cpu=pentium
// For now this is just a warning.
//@ build-pass
//@error-pattern: must be enabled

#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
