// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

struct Point {
    x: int,
    y: int,
}

impl Add<int,int> for Point {
    fn add(&self, z: &int) -> int {
        self.x + self.y + (*z)
    }
}

impl Point {
    pub fn times(&self, z: int) -> int {
        self.x * self.y * z
    }
}

fn a() {
    let mut p = Point {x: 3, y: 4};

    // ok (we can loan out rcvr)
    p + 3;
    p.times(3);
}

fn b() {
    let mut p = Point {x: 3, y: 4};

    // Here I create an outstanding loan and check that we get conflicts:

    let q = &mut p;

    p + 3;  //~ ERROR cannot borrow `p`
    p.times(3); //~ ERROR cannot borrow `p`

    *q + 3; // OK to use the new alias `q`
    q.x += 1; // and OK to mutate it
}

fn main() {
}
