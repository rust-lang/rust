// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics, volatile)]

use std::intrinsics::{volatile_load, volatile_store};
use std::ptr::{read_volatile, write_volatile};

pub fn main() {
    unsafe {
        let mut i : isize = 1;
        volatile_store(&mut i, 2);
        assert_eq!(volatile_load(&i), 2);
    }
    unsafe {
        let mut i : isize = 1;
        write_volatile(&mut i, 2);
        assert_eq!(read_volatile(&i), 2);
    }
}
