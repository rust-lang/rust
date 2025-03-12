//@ compile-flags: --target=x86_64-unknown-none --crate-type=lib
//@ needs-llvm-components: x86
//@ build-pass
#![feature(no_core, lang_items, x87_target_feature, const_trait_impl)]
#![no_core]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
#[const_trait]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
pub trait Sized: MetaSized {}

#[target_feature(enable = "x87")]
pub unsafe fn my_fun() {}
