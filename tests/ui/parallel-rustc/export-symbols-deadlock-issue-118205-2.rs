// Test for #118205, which causes a deadlock bug
//
//@ compile-flags:-C extra-filename=-1 -Z threads=16
//@ no-prefer-dynamic
//@ build-pass
//@ compare-output-by-lines

#![crate_name = "crateresolve1"]
#![crate_type = "lib"]

pub fn f() -> isize { 10 }
