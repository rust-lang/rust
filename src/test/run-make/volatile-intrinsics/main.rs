// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::intrinsics::{volatile_load, volatile_store};

pub fn main() {
    unsafe {
        let mut i : int = 1;
        volatile_store(&mut i, 2);
        assert_eq!(volatile_load(&i), 2);
    }
}
