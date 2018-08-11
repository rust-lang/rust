// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This should never be allowed -- since `x` is not `mut`, so `x.0`
// cannot be assigned twice.

#![feature(nll)]

fn var_then_field() {
    let x: (u32, u32);
    x = (22, 44);
    x.0 = 1; //~ ERROR
}

fn same_field_twice() {
    let x: (u32, u32);
    x.0 = 1; //~ ERROR
    x.0 = 22; //~ ERROR
    x.1 = 44; //~ ERROR
}

fn main() { }
