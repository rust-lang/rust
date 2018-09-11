// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Helper definition for test/run-pass/check-static-recursion-foreign.rs.

#![feature(libc)]

#![crate_name = "check_static_recursion_foreign_helper"]
#![crate_type = "lib"]

extern crate libc;

#[no_mangle]
pub static test_static: libc::c_int = 0;
