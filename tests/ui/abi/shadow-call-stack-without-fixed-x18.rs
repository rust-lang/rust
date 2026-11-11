//@ compile-flags: --target aarch64-unknown-none -Zsanitizer=shadow-call-stack
//@ dont-check-compiler-stderr
//@ needs-llvm-components: aarch64

#![allow(internal_features)]
#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[no_mangle]
pub fn foo() {}

//~? ERROR shadow-call-stack sanitizer is not supported for this target
