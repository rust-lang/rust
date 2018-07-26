// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:fat_drop.rs

extern crate fat_drop;

fn main() {
    unsafe {
        let data: &mut [u8] = &mut [0];
        let s: &mut fat_drop::S = std::mem::transmute::<&mut [u8], _>(data);
        std::ptr::drop_in_place(s);
        assert!(fat_drop::DROPPED);
    }
}
