// Regression test for MSVC link.exe failing to export weak definitions from dlls.
// See https://github.com/rust-lang/rust/pull/142568

//@ build-pass
//@ only-msvc
//@ revisions: link_exe lld
//@[lld] needs-rust-lld
//@[link_exe] compile-flags: -Zunstable-options -Clink-self-contained=-linker -Zlinker-features=-lld
//@[lld] compile-flags: -Zunstable-options -Clink-self-contained=+linker -Zlinker-features=+lld

#![feature(linkage)]
#![crate_type = "cdylib"]

#[linkage = "weak"]
#[no_mangle]
pub fn weak_function() {}

#[linkage = "weak"]
#[no_mangle]
pub static WEAK_STATIC: u8 = 42;
