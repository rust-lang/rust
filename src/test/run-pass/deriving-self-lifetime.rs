// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test FIXME #11820: & is unreliable in deriving

#[deriving(Eq,Ord)]
struct A<'a> {
    x: &'a int
}

pub fn main() {
    let a = A { x: &1 };
    let b = A { x: &2 };

    assert_eq!(a, a);
    assert_eq!(b, b);


    fail_unless!(a < b);
    fail_unless!(b > a);

    fail_unless!(a <= b);
    fail_unless!(a <= a);
    fail_unless!(b <= b);

    fail_unless!(b >= a);
    fail_unless!(b >= b);
    fail_unless!(a >= a);
}
