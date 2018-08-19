// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for #49257:
// emits good diagnostics for `..` pattern fragments not in the last position.

#![allow(unused)]

struct Point { x: u8, y: u8 }

fn main() {
    let p = Point { x: 0, y: 0 };
    let Point { .., y, } = p; //~ ERROR expected `}`, found `,`
    let Point { .., y } = p; //~ ERROR expected `}`, found `,`
    let Point { .., } = p; //~ ERROR expected `}`, found `,`
    let Point { .. } = p;
}
