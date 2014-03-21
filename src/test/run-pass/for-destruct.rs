// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Pair { x: int, y: int }

pub fn main() {
    for elt in (vec!(Pair {x: 10, y: 20}, Pair {x: 30, y: 0})).iter() {
        assert_eq!(elt.x + elt.y, 30);
    }
}
