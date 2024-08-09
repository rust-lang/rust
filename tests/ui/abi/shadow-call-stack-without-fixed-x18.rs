//@ compile-flags: --target aarch64-unknown-none -Zsanitizer=shadow-call-stack
//@ error-pattern: shadow-call-stack sanitizer is not supported for this target
//@ dont-check-compiler-stderr
//@ needs-llvm-components: aarch64

#![allow(internal_features)]
#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[no_mangle]
pub fn foo() {}
