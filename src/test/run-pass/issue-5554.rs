// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

use std::default::Default;

pub struct X<T> {
    a: T,
}

// reordering these bounds stops the ICE
//
// nmatsakis: This test used to have the bounds Default + PartialEq +
// Default, but having duplicate bounds became illegal.
impl<T: Default + PartialEq> Default for X<T> {
    fn default() -> X<T> {
        X { a: Default::default() }
    }
}

macro_rules! constants {
    () => {
        let _ : X<int> = Default::default();
    }
}

pub fn main() {
    constants!();
}
