// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::str;

static A: [u8, ..2] = ['h' as u8, 'i' as u8];
static B: &'static [u8, ..2] = &A;
static C: *u8 = B as *u8;

pub fn main() {
    unsafe {
        let foo = &A as *u8;
        fail_unless_eq!(str::raw::from_utf8(A), "hi");
        fail_unless_eq!(str::raw::from_buf_len(foo, A.len()), ~"hi");
        fail_unless_eq!(str::raw::from_buf_len(C, B.len()), ~"hi");
        fail_unless!(*C == A[0]);
        fail_unless!(*(&B[0] as *u8) == A[0]);

        let bar = str::raw::from_utf8(A).to_c_str();
        fail_unless_eq!(bar.with_ref(|buf| str::raw::from_c_str(buf)), ~"hi");
    }
}
