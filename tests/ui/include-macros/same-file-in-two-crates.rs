// This test makes sure that the compiler can handle the same source file to be
// part of the local crate *and* an upstream crate. This can happen, for example,
// when there is some auto-generated code that is part of both a library and an
// accompanying integration test.
//
// The test uses include!() to include a source file that is also part of
// an upstream crate.
//
// This is a regression test for https://github.com/rust-lang/rust/issues/85955.

//@ check-pass
//@ compile-flags: --crate-type=rlib
//@ aux-build:same-file-in-two-crates-aux.rs
extern crate same_file_in_two_crates_aux;

pub fn foo() -> u32 {
    same_file_in_two_crates_aux::some_function() +
    some_function()
}

include!("./auxiliary/same-file-in-two-crates-aux.rs");
