// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test related to #22779. In this case, the impl is an inherent impl,
// so it doesn't have to match any trait, so no error results.

#![feature(rustc_attrs)]
#![allow(dead_code)]

struct MySlice<'a, T:'a>(&'a mut [T]);

impl<'a, T> MySlice<'a, T> {
    fn renew<'b: 'a>(self) -> &'b mut [T] where 'a: 'b {
        &mut self.0[..]
    }
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
