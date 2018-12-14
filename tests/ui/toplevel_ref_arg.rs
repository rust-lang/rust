// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all)]
#![allow(unused)]

fn the_answer(ref mut x: u8) {
    *x = 42;
}

fn main() {
    let mut x = 0;
    the_answer(x);
    // Closures should not warn
    let y = |ref x| println!("{:?}", x);
    y(1u8);

    let ref x = 1;

    let ref y: (&_, u8) = (&1, 2);

    let ref z = 1 + 2;

    let ref mut z = 1 + 2;

    let (ref x, _) = (1, 2); // okay, not top level
    println!("The answer is {}.", x);
}
