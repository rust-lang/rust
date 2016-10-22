// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: make compiletest understand ignore-64bit / ignore-32bit

#![allow(unused)]
#![deny(nonportable_32_64)]

#[cfg(target_pointer_width = "64")]
fn check64() {
    let _: usize = 0u64.into();
    //~^ ERROR conversion `u64` -> `usize` is not portable between 64-bit and 32-bit platforms
    let _: usize = Into::into(0u64);
    //~^ ERROR conversion `u64` -> `usize` is not portable between 64-bit and 32-bit platforms
    let _: usize = From::from(0u64);
    //~^ ERROR conversion `u64` -> `usize` is not portable between 64-bit and 32-bit platforms

    let _: isize = 0i64.into();
    //~^ ERROR conversion `i64` -> `isize` is not portable between 64-bit and 32-bit platforms
    let _: isize = Into::into(0i64);
    //~^ ERROR conversion `i64` -> `isize` is not portable between 64-bit and 32-bit platforms
    let _: isize = From::from(0i64);
    //~^ ERROR conversion `i64` -> `isize` is not portable between 64-bit and 32-bit platforms

    let _: isize = 0u32.into();
    //~^ ERROR conversion `u32` -> `isize` is not portable between 64-bit and 32-bit platforms
    let _: isize = Into::into(0u32);
    //~^ ERROR conversion `u32` -> `isize` is not portable between 64-bit and 32-bit platforms
    let _: isize = From::from(0u32);
    //~^ ERROR conversion `u32` -> `isize` is not portable between 64-bit and 32-bit platforms
}

#[cfg(target_pointer_width = "32")]
fn check32() {
    let _: u32 = 0usize.into();
    // ERROR conversion `usize` -> `u32` is not portable between 64-bit and 32-bit platforms
    let _: u32 = Into::into(0usize);
    // ERROR conversion `usize` -> `u32` is not portable between 64-bit and 32-bit platforms
    let _: u32 = From::from(0usize);
    // ERROR conversion `usize` -> `u32` is not portable between 64-bit and 32-bit platforms

    let _: i32 = 0isize.into();
    // ERROR conversion `isize` -> `i32` is not portable between 64-bit and 32-bit platforms
    let _: i32 = Into::into(0isize);
    // ERROR conversion `isize` -> `i32` is not portable between 64-bit and 32-bit platforms
    let _: i32 = From::from(0isize);
    // ERROR conversion `isize` -> `i32` is not portable between 64-bit and 32-bit platforms

    let _: i64 = 0usize.into();
    // ERROR conversion `usize` -> `i64` is not portable between 64-bit and 32-bit platforms
    let _: i64 = Into::into(0usize);
    // ERROR conversion `usize` -> `i64` is not portable between 64-bit and 32-bit platforms
    let _: i64 = From::from(0usize);
    // ERROR conversion `usize` -> `i64` is not portable between 64-bit and 32-bit platforms
}

fn main() {}
