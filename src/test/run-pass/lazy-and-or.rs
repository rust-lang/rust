// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn incr(x: &mut int) -> bool { *x += 1; assert!((false)); return false; }

pub fn main() {
    let x = 1 == 2 || 3 == 3;
    assert!((x));
    let mut y: int = 10;
    info2!("{:?}", x || incr(&mut y));
    assert_eq!(y, 10);
    if true && x { assert!((true)); } else { assert!((false)); }
}
