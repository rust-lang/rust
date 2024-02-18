//@ compile-flags:-C extra-filename=-1 --emit=metadata
#![crate_name = "crateresolve2"]
#![crate_type = "lib"]

pub fn f() -> isize { 10 }
