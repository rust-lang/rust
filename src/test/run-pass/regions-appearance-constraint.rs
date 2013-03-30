// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Tests conditional rooting of the box y */

fn testfn(cond: bool) {
    let mut x = @3;
    let mut y = @4;

    let mut a = &*x;

    let mut exp = 3;
    if cond {
        a = &*y;

        exp = 4;
    }

    x = @5;
    y = @6;
    assert!(*a == exp);
}

pub fn main() {
}
