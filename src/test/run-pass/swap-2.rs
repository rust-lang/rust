// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn swap<T>(v: &[mut T], i: int, j: int) { v[i] <-> v[j]; }

fn main() {
    let mut a: ~[int] = ~[0, 1, 2, 3, 4, 5, 6];
    swap(a, 2, 4);
    assert (a[2] == 4);
    assert (a[4] == 2);
    let mut n = 42;
    n <-> a[0];
    assert (a[0] == 42);
    assert (n == 0);
}
