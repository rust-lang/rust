// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd, test)]

extern crate test;

#[repr(simd)]
pub struct Mu64(pub u64, pub u64, pub u64, pub u64);

fn main() {
    // This ensures an unaligned pointer even in optimized builds, though LLVM
    // gets enough type information to actually not mess things up in that case,
    // but at the time of writing this, it's enough to trigger the bug in
    // non-optimized builds
    unsafe {
        let memory = &mut [0u64; 8] as *mut _ as *mut u8;
        let misaligned_ptr: &mut [u8; 32] = {
            std::mem::transmute(memory.offset(1))
        };
        *misaligned_ptr = std::mem::transmute(Mu64(1, 1, 1, 1));
        test::black_box(memory);
    }
}
