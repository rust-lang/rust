// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to reinitilize box with moved referent
#![feature(nll)]
static mut ORDER: [usize; 3] = [0, 0, 0];
static mut INDEX: usize = 0;

struct Dropee (usize);

impl Drop for Dropee {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = self.0;
            INDEX = INDEX + 1;
        }
    }
}

fn add_sentintel() {
    unsafe {
        ORDER[INDEX] = 2;
        INDEX = INDEX + 1;
    }
}

fn main() {
    let mut x = Box::new(Dropee(1));
    *x;  // move out from `*x`
    add_sentintel();
    *x = Dropee(3); // re-initialize `*x`
    {x}; // drop value
    unsafe {
        assert_eq!(ORDER, [1, 2, 3]);
    }
}
