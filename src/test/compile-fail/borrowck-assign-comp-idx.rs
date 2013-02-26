// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type point = { x: int, y: int };

fn a() {
    let mut p = ~[1];

    // Create an immutable pointer into p's contents:
    let _q: &int = &p[0]; //~ NOTE loan of mutable vec content granted here

    p[0] = 5; //~ ERROR assigning to mutable vec content prohibited due to outstanding loan
}

fn borrow(_x: &[int], _f: fn()) {}

fn b() {
    // here we alias the mutable vector into an imm slice and try to
    // modify the original:

    let mut p = ~[1];

    do borrow(p) { //~ NOTE loan of mutable vec content granted here
        p[0] = 5; //~ ERROR assigning to mutable vec content prohibited due to outstanding loan
    }
}

fn c() {
    // Legal because the scope of the borrow does not include the
    // modification:
    let mut p = ~[1];
    borrow(p, ||{});
    p[0] = 5;
}

fn main() {
}

