// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

trait double {
    fn double(self) -> uint;
}

impl double for uint {
    fn double(self) -> uint { self }
}

impl double for Box<uint> {
    fn double(self) -> uint { *self * 2_usize }
}

pub fn main() {
    let x = box 3_usize;
    assert_eq!(x.double(), 6_usize);
}
