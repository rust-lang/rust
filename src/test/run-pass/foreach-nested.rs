// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn two(it: |int|) { it(0); it(1); }

pub fn main() {
    let mut a: Vec<int> = vec!(-1, -1, -1, -1);
    let mut p: int = 0;
    two(|i| {
        two(|j| { *a.get_mut(p as uint) = 10 * i + j; p += 1; })
    });
    assert_eq!(*a.get(0), 0);
    assert_eq!(*a.get(1), 1);
    assert_eq!(*a.get(2), 10);
    assert_eq!(*a.get(3), 11);
}
