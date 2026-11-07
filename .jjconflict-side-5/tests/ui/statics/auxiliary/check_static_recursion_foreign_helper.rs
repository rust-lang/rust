// Helper definition for test/run-pass/check-static-recursion-foreign.rs.

#![feature(rustc_private)]

#![crate_name = "check_static_recursion_foreign_helper"]
#![crate_type = "lib"]

use std::ffi::c_int;

#[no_mangle]
pub static test_static: c_int = 0;
