//@ compile-flags:-C extra-filename=-1 -Z threads=16
//@ no-prefer-dynamic
//@ build-pass
#![crate_name = "crateresolve1"]
#![crate_type = "lib"]

pub fn f() -> isize { 10 }
