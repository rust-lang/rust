// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq,Ord)]
struct A<'self> {
    x: &'self int
}

fn main() {
    let a = A { x: &1 };
    let b = A { x: &2 };

    assert_eq!(a, a);
    assert_eq!(b, b);


    assert!(a < b);
    assert!(b > a);

    assert!(a <= b);
    assert!(a <= a);
    assert!(b <= b);

    assert!(b >= a);
    assert!(b >= b);
    assert!(a >= a);
}
