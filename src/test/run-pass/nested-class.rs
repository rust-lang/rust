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
    struct b {
        i: int,
    }

    pub impl b {
        fn do_stuff(&self) -> int { return 37; }
    }

    fn b(i:int) -> b {
        b {
            i: i
        }
    }

    //  fn b(x:int) -> int { fail!(); }

    let z = b(42);
    assert_eq!(z.i, 42);
    assert_eq!(z.do_stuff(), 37);
}
