// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
struct Thing {
    x: int
}

impl Mul<int, Thing>*/ for Thing/* { //~ ERROR Look ma, no Mul!
    fn mul(c: &int) -> Thing {
        Thing {x: self.x * *c}
    }
}

fn main() {
    let u = Thing {x: 2};
    let _v = u.mul(&3); // Works
    let w = u * 3; // Works!!
    io::println(fmt!("%i", w.x));

    /*
    // This doesn't work though.
    let u2 = u as @Mul<int, Thing>;
    let w2 = u2.mul(&4);
    io::println(fmt!("%i", w2.x));
    */
}
