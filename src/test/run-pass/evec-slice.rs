// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let x : &[int] = &[1,2,3,4,5];
    let mut z = &[1,2,3,4,5];
    z = x;
    assert!(z[0] == 1);
    assert!(z[4] == 5);

    let a : &[int] = &[1,1,1,1,1];
    let b : &[int] = &[2,2,2,2,2];
    let c : &[int] = &[2,2,2,2,3];
    let cc : &[int] = &[2,2,2,2,2,2];

    debug!(a);

    assert!(a < b);
    assert!(a <= b);
    assert!(a != b);
    assert!(b >= a);
    assert!(b > a);

    debug!(b);

    assert!(b < c);
    assert!(b <= c);
    assert!(b != c);
    assert!(c >= b);
    assert!(c > b);

    assert!(a < c);
    assert!(a <= c);
    assert!(a != c);
    assert!(c >= a);
    assert!(c > a);

    debug!(c);

    assert!(a < cc);
    assert!(a <= cc);
    assert!(a != cc);
    assert!(cc >= a);
    assert!(cc > a);

    debug!(cc);
}
