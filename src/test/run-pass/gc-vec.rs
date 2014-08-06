// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::gc::{GC};

fn main() {
    // A fixed-size array allocated in a garbage-collected box
    let x = box(GC) [1i, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(x[0], 1);
    assert_eq!(x[6], 7);
    assert_eq!(x[9], 10);

    let y = x;
    assert!(*y == [1i, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}
