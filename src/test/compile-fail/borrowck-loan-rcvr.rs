// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

struct point { x: int, y: int }

trait methods {
    fn impurem(&self);
    fn blockm(&self, f: ||);
}

impl methods for point {
    fn impurem(&self) {
    }

    fn blockm(&self, f: ||) { f() }
}

fn a() {
    let mut p = point {x: 3, y: 4};

    // Here: it's ok to call even though receiver is mutable, because we
    // can loan it out.
    p.impurem();

    // But in this case we do not honor the loan:
    p.blockm(|| {
        p.x = 10; //~ ERROR cannot assign
    })
}

fn b() {
    let mut p = point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let l = &mut p;
    p.impurem(); //~ ERROR cannot borrow

    l.x += 1;
}

fn main() {
}
