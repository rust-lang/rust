// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that type inference for range patterns works correctly (is bi-directional).

pub fn main() {
    match 1 {
        1 ... 3 => {}
        _ => panic!("should match range")
    }
    match 1 {
        1 ... 3u16 => {}
        _ => panic!("should match range with inferred start type")
    }
    match 1 {
        1u16 ... 3 => {}
        _ => panic!("should match range with inferred end type")
    }
}
