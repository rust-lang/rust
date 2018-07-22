// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let tiles = Default::default();
    for row in &mut tiles {
        for tile in row {
            //~^ NOTE the element type for this iterator is not specified
            *tile = 0;
            //~^ ERROR type annotations needed
            //~| NOTE cannot infer type
            //~| NOTE type must be known at this point
        }
    }

    let tiles: [[usize; 3]; 3] = tiles;
}
