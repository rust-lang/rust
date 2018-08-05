// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that liveness due to a closure capture gives a special note

#![feature(nll)]

fn use_as_borrow_capture(mut x: i32) {
    let y = &x;
    x = 0; //~ ERROR
    || *y;
}

fn use_as_borrow_mut_capture(mut x: i32) {
    let y = &mut x;
    x = 0; //~ ERROR
    || *y = 1;
}

fn use_as_move_capture(mut x: i32) {
    let y = &x;
    x = 0; //~ ERROR
    move || *y;
}

fn main() {}
