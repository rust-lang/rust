// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that having something of type ! doesn't screw up type-checking and that it coerces to the
// LUB type of the other match arms.

fn main() {
    let v: Vec<u32> = Vec::new();
    match 0u32 {
        0 => &v,
        1 => return,
        _ => &v[..],
    };
}

