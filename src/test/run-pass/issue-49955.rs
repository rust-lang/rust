// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=compare

const ALL_THE_NUMS: [u32; 1] = [
    1
];

#[inline(never)]
fn array(i: usize) -> &'static u32 {
    return &ALL_THE_NUMS[i];
}

#[inline(never)]
fn tuple_field() -> &'static u32 {
    &(42,).0
}

fn main() {
    assert_eq!(tuple_field().to_string(), "42");
    assert_eq!(array(0).to_string(), "1");
}
