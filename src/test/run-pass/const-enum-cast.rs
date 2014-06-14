// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum A { A1, A2 }
enum B { B1=0, B2=2 }

pub fn main () {
    static c1: int = A2 as int;
    static c2: int = B2 as int;
    let a1 = A2 as int;
    let a2 = B2 as int;
    assert_eq!(c1, 1);
    assert_eq!(c2, 2);
    assert_eq!(a1, 1);
    assert_eq!(a2, 2);
}
