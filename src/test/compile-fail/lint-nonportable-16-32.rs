// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]
#![deny(nonportable_16_32)]

fn check() {
    let _: usize = 0u32.into();
    //~^ ERROR conversion `u32` -> `usize` is not portable between 32-bit and 16-bit platforms
    let _: usize = Into::into(0u32);
    //~^ ERROR conversion `u32` -> `usize` is not portable between 32-bit and 16-bit platforms
    let _: usize = From::from(0u32);
    //~^ ERROR conversion `u32` -> `usize` is not portable between 32-bit and 16-bit platforms

    let _: isize = 0i32.into();
    //~^ ERROR conversion `i32` -> `isize` is not portable between 32-bit and 16-bit platforms
    let _: isize = Into::into(0i32);
    //~^ ERROR conversion `i32` -> `isize` is not portable between 32-bit and 16-bit platforms
    let _: isize = From::from(0i32);
    //~^ ERROR conversion `i32` -> `isize` is not portable between 32-bit and 16-bit platforms

    let _: isize = 0u16.into();
    //~^ ERROR conversion `u16` -> `isize` is not portable between 32-bit and 16-bit platforms
    let _: isize = Into::into(0u16);
    //~^ ERROR conversion `u16` -> `isize` is not portable between 32-bit and 16-bit platforms
    let _: isize = From::from(0u16);
    //~^ ERROR conversion `u16` -> `isize` is not portable between 32-bit and 16-bit platforms
}

fn main() {}
