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
    let x = &"hello";
    let v = &"hello";
    let mut y : &str = &"there";

    info!(x);
    info!(y);

    assert_eq!(x[0], 'h' as u8);
    assert_eq!(x[4], 'o' as u8);

    let z : &str = &"thing";
    assert_eq!(v, x);
    assert!(x != z);

    let a = &"aaaa";
    let b = &"bbbb";

    let c = &"cccc";
    let cc = &"ccccc";

    info!(a);

    assert!(a < b);
    assert!(a <= b);
    assert!(a != b);
    assert!(b >= a);
    assert!(b > a);

    info!(b);

    assert!(a < c);
    assert!(a <= c);
    assert!(a != c);
    assert!(c >= a);
    assert!(c > a);

    info!(c);

    assert!(c < cc);
    assert!(c <= cc);
    assert!(c != cc);
    assert!(cc >= c);
    assert!(cc > c);

    info!(cc);
}
