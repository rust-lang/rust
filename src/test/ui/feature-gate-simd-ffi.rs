// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd)]
#![allow(dead_code)]

#[repr(simd)]
#[derive(Copy, Clone)]
#[repr(C)]
struct LocalSimd(u8, u8);

extern {
    fn baz() -> LocalSimd; //~ ERROR use of SIMD type
    fn qux(x: LocalSimd); //~ ERROR use of SIMD type
}

fn main() {}
