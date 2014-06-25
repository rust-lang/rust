// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(intrinsics)]

use std::mem::transmute;

mod rusti {
    extern "rust-intrinsic" {
        pub fn init<T>() -> T;
        pub fn move_val_init<T>(dst: &mut T, src: T);
    }
}

pub fn main() {
    unsafe {
        let x = box 1i;
        let mut y = rusti::init();
        let mut z: *const uint = transmute(&x);
        rusti::move_val_init(&mut y, x);
        assert_eq!(*y, 1);
        assert_eq!(*z, 0); // `x` is nulled out, not directly visible
    }
}
