// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod _common;

use _common::validate;

fn main() {
    // Skip e = 0 because small-u32 already does those.
    for e in 1..301 {
        for i in 0..10000 {
            // If it ends in zeros, the parser will strip those (and adjust the exponent),
            // which almost always (except for exponents near +/- 300) result in an input
            // equivalent to something we already generate in a different way.
            if i % 10 == 0 {
                continue;
            }
            validate(format!("{}e{}", i, e));
            validate(format!("{}e-{}", i, e));
        }
    }
}
