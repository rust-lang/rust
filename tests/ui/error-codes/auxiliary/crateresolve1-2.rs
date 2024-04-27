//@ compile-flags:-C extra-filename=-2
//@ no-prefer-dynamic
#![crate_name = "crateresolve1"]
#![crate_type = "lib"]

pub fn f() -> isize { 20 }
