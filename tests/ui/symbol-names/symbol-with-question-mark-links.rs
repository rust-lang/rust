// This test ensures functions with an exported name beginning with a question mark
// successfully compile and link.
//
// Regression test for <https://github.com/rust-lang/rust/issues/44282>

//@ build-pass
//@ only-windows
//@ only-x86
// Reason: This test regards a linker issue which only applies to Windows.
// Specifically, it only occurs due to Windows x86 name decoration, combined with
// a mismatch between LLVM's decoration logic and Rust's (for `lib.def` generation)

#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn decorated(a: i32, b: i32) -> i32 {
    1
}

// This isn't just `?undecorated` because MSVC's linker fails if the decorated
// symbol is not valid.
#[export_name = "?undecorated@@YAXXZ"]
pub extern "C" fn undecorated(a: i32, b: i32) -> i32 {
    2
}
