// Helper definition for test/run-pass/check-static-recursion-foreign.rs.

#![feature(rustc_private)]

#![crate_name = "check_static_recursion_foreign_helper"]
#![crate_type = "lib"]

extern crate libc;

#[no_mangle]
pub static test_static: libc::c_int = 0;
