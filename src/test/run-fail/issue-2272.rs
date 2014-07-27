// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// error-pattern:explicit failure
// Issue #2272 - unwind this without leaking the unique pointer

use std::gc::{Gc, GC};

struct X { y: Y, a: Box<int> }

struct Y { z: Gc<int> }

fn main() {
    let _x = X {
        y: Y {
            z: box(GC) 0
        },
        a: box 0
    };
    fail!();
}
