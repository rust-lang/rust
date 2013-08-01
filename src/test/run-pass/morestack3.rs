// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test newsched transition

// Here we're testing that all of the argument registers, argument
// stack slots, and return value are preserved across split stacks.
fn getbig(a0: int,
          a1: int,
          a2: int,
          a3: int,
          a4: int,
          a5: int,
          a6: int,
          a7: int,
          a8: int,
          a9: int) -> int {

    assert_eq!(a0 + 1, a1);
    assert_eq!(a1 + 1, a2);
    assert_eq!(a2 + 1, a3);
    assert_eq!(a3 + 1, a4);
    assert_eq!(a4 + 1, a5);
    assert_eq!(a5 + 1, a6);
    assert_eq!(a6 + 1, a7);
    assert_eq!(a7 + 1, a8);
    assert_eq!(a8 + 1, a9);
    if a0 != 0 {
        let j = getbig(a0 - 1,
                       a1 - 1,
                       a2 - 1,
                       a3 - 1,
                       a4 - 1,
                       a5 - 1,
                       a6 - 1,
                       a7 - 1,
                       a8 - 1,
                       a9 - 1);
        assert_eq!(j, a0 - 1);
    }
    return a0;
}

pub fn main() {
    let a = 1000;
    getbig(a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9);
}
