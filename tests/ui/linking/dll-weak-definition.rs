// Regression test for MSVC link.exe failing to export weak definitions from dlls.
// See https://github.com/rust-lang/rust/pull/142568

//@ build-pass
//@ only-msvc
//@ revisions: link_exe lld
//@[link_exe] compile-flags: -Clinker=link.exe
//@[lld] needs-rust-lld
//@[lld] compile-flags: -Clinker=rust-lld

#![feature(linkage)]
#![crate_type = "cdylib"]

#[linkage = "weak"]
#[no_mangle]
pub fn weak_function() {}

#[linkage = "weak"]
#[no_mangle]
pub static WEAK_STATIC: u8 = 42;
