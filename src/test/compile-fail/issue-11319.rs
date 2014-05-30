// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    match Some(10) {
    //~^ ERROR match arms have incompatible types: expected `bool` but found `()`
        Some(5) => false,
        Some(2) => true,
        None    => (), //~ NOTE match arm with an incompatible type
        _       => true
    }
}
