// compile-flags: -Clinker-plugin-lto -Copt-level=s
// build-pass
// no-prefer-dynamic

#![crate_type = "rlib"]

pub fn foo() {}
