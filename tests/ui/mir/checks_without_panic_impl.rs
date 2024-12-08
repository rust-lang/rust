// Ensures that the alignment check we insert for raw pointer dereferences
// does not prevent crates without a panic_impl from compiling.
// See rust-lang/rust#109996

//@ build-pass
//@ compile-flags: -Cdebug-assertions=yes

#![crate_type = "lib"]

#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

#[lang = "sized"]
trait Foo {}

pub unsafe fn foo(x: *const i32) -> &'static i32 { unsafe { &*x } }
