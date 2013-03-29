// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use std::arc;

fn main() {
    let v = ~[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = arc::ARC(v);

    do task::spawn() { //~ NOTE `arc_v` moved into closure environment here
        let v = *arc::get(&arc_v);
        assert!(v[3] == 4);
    };

    assert!((*arc::get(&arc_v))[2] == 3); //~ ERROR use of moved value: `arc_v`

    info!(arc_v);
}
