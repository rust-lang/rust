// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="cdylib"]

extern crate an_rlib;

// This should not be exported
pub fn public_rust_function_from_cdylib() {}

// This should be exported
#[no_mangle]
pub extern "C" fn public_c_function_from_cdylib() {
    an_rlib::public_c_function_from_rlib();
}
