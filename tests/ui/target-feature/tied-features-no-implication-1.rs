//@ revisions: paca pacg
//@ compile-flags: --crate-type=rlib --target=aarch64-unknown-linux-gnu
//@ needs-llvm-components: aarch64
//@[paca] compile-flags: -Ctarget-feature=+paca
//@[paca] error-pattern: the target features paca, pacg must all be either enabled or disabled together
//@[pacg] compile-flags: -Ctarget-feature=+pacg
//@[paca] error-pattern: the target features paca, pacg must all be either enabled or disabled together
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

// In this test, demonstrate that +paca and +pacg both result in the tied feature error if there
// isn't something causing an error.
// See tied-features-no-implication.rs

#[cfg(target_feature = "pacg")]
pub unsafe fn foo() {
}

//~? ERROR the target features paca, pacg must all be either enabled or disabled together
