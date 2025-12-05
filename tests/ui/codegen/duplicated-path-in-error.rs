//@ revisions: musl gnu
//@ only-linux
//@ ignore-cross-compile because this relies on host libc behaviour
//@ compile-flags: -Zcodegen-backend=/non-existing-one.so
//@[gnu] only-gnu
//@[musl] only-musl

// This test ensures that the error of the "not found dylib" doesn't duplicate
// the path of the dylib.
//
// glibc and musl have different dlopen error messages, so the expected error
// message differs between the two.

fn main() {}

//~? ERROR couldn't load codegen backend /non-existing-one.so
