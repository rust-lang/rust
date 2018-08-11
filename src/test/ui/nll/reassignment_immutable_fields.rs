// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is currently disallowed, but we hope someday to support it.
//
// FIXME(#21232)

#![feature(nll)]

fn assign_both_fields_and_use() {
    let x: (u32, u32);
    x.0 = 1; //~ ERROR
    x.1 = 22; //~ ERROR
    drop(x.0);
    drop(x.1);
}

fn assign_both_fields_the_use_var() {
    let x: (u32, u32);
    x.0 = 1; //~ ERROR
    x.1 = 22; //~ ERROR
    drop(x); //~ ERROR
}

fn main() { }
