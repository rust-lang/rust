// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]
#![feature(intrinsics)]
// needed to check for drop fill word.
#![feature(filling_drop)]

use std::mem::{self, transmute};

mod rusti {
    extern "rust-intrinsic" {
        pub fn init<T>() -> T;
        pub fn move_val_init<T>(dst: &mut T, src: T);
    }
}

pub fn main() {
    unsafe {
        let x: Box<_> = box 1;
        let mut y = rusti::init();
        let mut z: *const uint = transmute(&x);
        rusti::move_val_init(&mut y, x);
        assert_eq!(*y, 1);
        // `x` is nulled out, not directly visible
        assert_eq!(*z, mem::POST_DROP_USIZE);
    }
}
