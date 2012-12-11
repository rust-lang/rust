// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Point { 
    x: int,
    y: int,
}

impl Point : ops::Add<int,int> {
    pure fn add(&self, z: &int) -> int {
        self.x + self.y + (*z)
    }
}

impl Point {
    fn times(z: int) -> int {
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

    let q = &mut p; //~ NOTE prior loan as mutable granted here

    p + 3;  // ok for pure fns
    p.times(3); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan

    q.x += 1;
}

fn c() {
    // Here the receiver is in aliased memory and hence we cannot
    // consider it immutable:
    let q = @mut Point {x: 3, y: 4};

    // ...this is ok for pure fns
    *q + 3;


    // ...but not impure fns
    (*q).times(3); //~ ERROR illegal borrow unless pure
    //~^ NOTE impure due to access to impure function
}

fn main() {
}

