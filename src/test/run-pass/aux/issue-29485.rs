// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="a"]
#![crate_type = "lib"]

pub struct X(pub u8);

impl Drop for X {
    fn drop(&mut self) {
        assert_eq!(self.0, 1)
    }
}

pub fn f(x: &mut X, g: fn()) {
    x.0 = 1;
    g();
    x.0 = 0;
}
