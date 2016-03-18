// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that fn item types are zero-sized.

use std::mem::{size_of, size_of_val};

fn main() {
    assert_eq!(size_of_val(&main), 0);

    let (a, b) = (size_of::<u8>, size_of::<u16>);
    assert_eq!(size_of_val(&a), 0);
    assert_eq!(size_of_val(&b), 0);
    assert_eq!((a(), b()), (1, 2));
}
