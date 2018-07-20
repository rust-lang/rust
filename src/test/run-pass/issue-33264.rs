// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, non_upper_case_globals)]
#![feature(asm)]

#[repr(C)]
pub struct D32x4(f32,f32,f32,f32);

impl D32x4 {
    fn add(&self, vec: Self) -> Self {
        unsafe {
            let ret: Self;
            asm!("
                 movaps $1, %xmm1
                 movaps $2, %xmm2
                 addps %xmm1, %xmm2
                 movaps $xmm1, $0
                 "
                 : "=r"(ret)
                 : "1"(self), "2"(vec)
                 : "xmm1", "xmm2"
                 );
            ret
        }
    }
}

fn main() { }

