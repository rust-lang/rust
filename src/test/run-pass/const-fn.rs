// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A very basic test of const fn functionality.

#![feature(const_fn)]

const fn add(x: u32, y: u32) -> u32 {
    x + y
}

const fn sub(x: u32, y: u32) -> u32 {
    x - y
}

const SUM: u32 = add(44, 22);
const DIFF: u32 = sub(44, 22);

fn main() {
    assert_eq!(SUM, 66);
    assert!(SUM != 88);

    assert_eq!(DIFF, 22);

    let _: [&'static str; sub(100, 99) as usize] = ["hi"];
}
