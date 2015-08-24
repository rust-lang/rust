// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

use std::cmp::PartialEq;

struct Input<'a> {
    _bytes: &'a [u8],
}

impl <'a> PartialEq for Input<'a> {
    fn eq(&self, _other: &Input<'a>) -> bool {
        panic!()
    }
}

struct Input2<'a> {
    i: Input<'a>
}

impl<'a, 'b> PartialEq<Input2<'b>> for Input2<'a> where Input<'a> : PartialEq<Input<'b>> {
    fn eq(&self, other: &Input2<'b>) -> bool {
        self.i == other.i
    }
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
