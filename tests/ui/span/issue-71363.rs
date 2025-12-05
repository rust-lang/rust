//@ compile-flags: -Z ui-testing=no

struct MyError;
impl std::error::Error for MyError {}
//~^ ERROR: `MyError` doesn't implement `std::fmt::Display`
//~| ERROR: `MyError` doesn't implement `Debug`

fn main() {}

// This test relies on library/std/src/error.rs *not* being included in the error message, so that
// we can test whether a file not included in the error message affects it (more specifically
// whether the line number of the excluded file affects the indentation of the other line numbers).
//
// To test this we're simulating a remap of the rust src base (so that library/std/src/error.rs
// does not point to a local file) *and* we're disabling the code to try mapping a remapped path to
// a local file (which would defeat the purpose of the former flag).
//
// Note that this comment is at the bottom of the file intentionally, as we need the line number of
// the impl to be lower than 10.
