// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #25180

const empty: &'static Fn() = &|| println!("ICE here");

const one_argument: &'static Fn(u32) = &|y| println("{}", y);

const plus_21: &'static Fn(u32) -> u32 = |y| y + 21;

const multi_and_local: &'static Fn(u32, u32) -> u32 = |x, y| {
    let tmp = x + y;
    tmp * 2;
};

pub fn main() {
    empty();
    one_argument(42);
    assert!(plus_21(21) == 42);
    assert!(multi_and_local(1, 2) == 6);
}

