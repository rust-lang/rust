// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{str, string};

const A: [u8; 2] = ['h' as u8, 'i' as u8];
const B: &'static [u8; 2] = &A;
const C: *const u8 = B as *const u8;

pub fn main() {
    unsafe {
        let foo = &A as *const u8;
        assert_eq!(foo, C);
        assert_eq!(str::from_utf8_unchecked(&A), "hi");
        assert_eq!(*C, A[0]);
        assert_eq!(*(&B[0] as *const u8), A[0]);
    }
}
