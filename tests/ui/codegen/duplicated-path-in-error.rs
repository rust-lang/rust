//@ only-linux
//@ compile-flags: -Zcodegen-backend=/non-existing-one.so

// This test ensures that the error of the "not found dylib" doesn't duplicate
// the path of the dylib.

fn main() {}

//~? ERROR couldn't load codegen backend /non-existing-one.so
