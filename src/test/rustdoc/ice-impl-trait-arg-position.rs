// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Read;

pub struct D<R>(R);

pub struct A;
impl A {
    pub fn a(&self, d: impl IntoIterator<Item = D<impl Read>>) -> Result<(), ()> {
        unimplemented!()
    }

    pub fn b(&self, d: impl IntoIterator<Item = D<()>>) -> Result<(), ()> {
        unimplemented!()
    }
}
