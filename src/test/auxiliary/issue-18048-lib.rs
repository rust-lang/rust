// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]
#![feature(associated_types)]

pub trait Bar {
    type T;

    fn get(x: Option<Self>) -> <Self as Bar>::T;
}

impl Bar for int {
    type T = uint;

    fn get(_: Option<int>) -> uint { 22 }
}
