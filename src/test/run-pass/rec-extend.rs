// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
type point = {x: int, y: int};

fn main() {
    let origin: point = {x: 0, y: 0};
    let right: point = {x: origin.x + 10,.. origin};
    let up: point = {y: origin.y + 10,.. origin};
    assert (origin.x == 0);
    assert (origin.y == 0);
    assert (right.x == 10);
    assert (right.y == 0);
    assert (up.x == 0);
    assert (up.y == 10);
}
