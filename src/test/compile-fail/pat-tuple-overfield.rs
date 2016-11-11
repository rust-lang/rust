// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S(u8, u8, u8);

fn main() {
    match (1, 2, 3) {
        (1, 2, 3, 4) => {} //~ ERROR mismatched types
        (1, 2, .., 3, 4) => {} //~ ERROR mismatched types
        _ => {}
    }
    match S(1, 2, 3) {
        S(1, 2, 3, 4) => {}
        //~^ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
        S(1, 2, .., 3, 4) => {}
        //~^ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
        _ => {}
    }
}
