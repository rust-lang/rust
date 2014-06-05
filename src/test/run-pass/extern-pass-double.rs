// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "rust_test_helpers")]
extern {
    pub fn rust_dbg_extern_identity_double(v: f64) -> f64;
}

pub fn main() {
    unsafe {
        assert_eq!(22.0_f64, rust_dbg_extern_identity_double(22.0_f64));
    }
}
