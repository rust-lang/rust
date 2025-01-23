//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ build-pass
#![feature(no_core, lang_items, x87_target_feature)]
#![no_core]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[target_feature(enable = "x87")]
pub unsafe fn my_fun() {}
