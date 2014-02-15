// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate extra;

use std::vec;

trait methods {
    fn to_bytes(&self) -> ~[u8];
}

impl methods for () {
    fn to_bytes(&self) -> ~[u8] {
        vec::from_elem(0, 0u8)
    }
}

// the position of this function is significant! - if it comes before methods
// then it works, if it comes after it then it doesn't!
fn to_bools(bitv: Storage) -> ~[bool] {
    vec::from_fn(8, |i| {
        let w = i / 64;
        let b = i % 64;
        let x = 1u64 & (bitv.storage[w] >> b);
        x == 1u64
    })
}

struct Storage { storage: ~[u64] }

pub fn main() {
    let bools = ~[false, false, true, false, false, true, true, false];
    let bools2 = to_bools(Storage{storage: ~[0b01100100]});

    for i in range(0u, 8) {
        println!("{} => {} vs {}", i, bools[i], bools2[i]);
    }

    assert_eq!(bools, bools2);
}
