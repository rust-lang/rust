// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name="cci_impl_lib", vers="0.0")];

trait uint_helpers {
    fn to(&self, v: uint, f: |uint|);
}

impl uint_helpers for uint {
    #[inline]
    fn to(&self, v: uint, f: |uint|) {
        let mut i = *self;
        while i < v {
            f(i);
            i += 1u;
        }
    }
}
