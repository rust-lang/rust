// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const C: u8 = 0; //~ NOTE a constant `C` is defined here

fn main() {
    match 1u8 {
        mut C => {} //~ ERROR match bindings cannot shadow constants
        //~^ NOTE cannot be named the same as a constant
        _ => {}
    }
}
