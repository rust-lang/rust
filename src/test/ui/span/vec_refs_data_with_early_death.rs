// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is a simple example of code that violates the dropck
// rules: it pushes `&x` and `&y` into `v`, but the referenced data
// will be dropped before the vector itself is.

// (In principle we know that `Vec` does not reference the data it
//  owns from within its drop code, apart from calling drop on each
//  element it owns; thus, for data like this, it seems like we could
//  loosen the restrictions here if we wanted. But it also is not
//  clear whether such loosening is terribly important.)

fn main() {
    let mut v = Vec::new();

    let x: i8 = 3;
    let y: i8 = 4;

    v.push(&x);
    v.push(&y);

    assert_eq!(v, [&3, &4]);
}
//~^ ERROR `x` does not live long enough
//~| ERROR `y` does not live long enough
