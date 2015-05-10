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

const EMPTY: &'static Fn() = &|| println!("ICE here");

const ONE_ARGUMENT: &'static Fn(u32) = &|y| println!("{}", y);

const PLUS_21: &'static (Fn(u32) -> u32) = &|y| y + 21;

const MULTI_AND_LOCAL: &'static (Fn(u32, u32) -> u32) = &|x, y| {
    let tmp = x + y;
    tmp * 2
};

pub fn main() {
    EMPTY();
    ONE_ARGUMENT(42);
    assert!(PLUS_21(21) == 42);
    assert!(MULTI_AND_LOCAL(1, 2) == 6);
}

