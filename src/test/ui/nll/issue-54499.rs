// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

fn foo() {
    let mut s: (i32, i32);
    s.0 = 1;
    s.1 = 2;
    println!("{} {}", s.0, s.1);
}

fn bar() {
    let s: (i32, i32);
    s.0 = 3;
    s.1 = 4;
    println!("{} {}", s.0, s.1);
}

fn main() {}
