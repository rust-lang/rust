// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test messages where a closure capture conflicts with itself because it's in
// a loop.

#![feature(nll)]

fn repreated_move(x: String) {
    for i in 0..10 {
        || x; //~ ERROR
    }
}

fn repreated_mut_borrow(mut x: String) {
    let mut v = Vec::new();
    for i in 0..10 {
        v.push(|| x = String::new()); //~ ERROR
    }
}

fn repreated_unique_borrow(x: &mut String) {
    let mut v = Vec::new();
    for i in 0..10 {
        v.push(|| *x = String::new()); //~ ERROR
    }
}

fn main() {}
