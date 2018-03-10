// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

// run-pass

fn vec() {
    let mut _x = vec!['c'];
    let _y = &_x;
    _x = Vec::new();
}

fn int() {
    let mut _x = 5;
    let _y = &_x;
    _x = 7;
}

fn main() {
    vec();
    int();
}
