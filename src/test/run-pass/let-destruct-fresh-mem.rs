// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let u = {x: 10, y: @{a: 20}};
    let mut {x: x, y: @{a: a}} = u;
    x = 100;
    a = 100;
    assert (x == 100);
    assert (a == 100);
    assert (u.x == 10);
    assert (u.y.a == 20);
}
