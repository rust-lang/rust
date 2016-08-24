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

#[inline(never)]
fn new(x: u64) -> Mu64 {
    Mu64(x, x, x, x)
}

#[inline(never)]
fn invoke_doom(x: &u8) -> [u8; 32] {
    // This transmute used to directly store the SIMD vector into a location
    // that isn't necessarily properly aligned
    unsafe { std::mem::transmute(new(*x as u64)) }
}

fn main() {
    // Try to get the dest for the invoke_doom calls to be misaligned even in optimized builds
    let x = 0;
    test::black_box(invoke_doom(&x));
    let y = 1;
    test::black_box(invoke_doom(&y));
}
