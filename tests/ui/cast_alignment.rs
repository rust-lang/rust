// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Test casts for alignment issues

#![feature(libc)]

extern crate libc;

#[warn(clippy::cast_ptr_alignment)]
#[allow(clippy::no_effect, clippy::unnecessary_operation, clippy::cast_lossless)]
fn main() {
    /* These should be warned against */

    // cast to more-strictly-aligned type
    (&1u8 as *const u8) as *const u16;
    (&mut 1u8 as *mut u8) as *mut u16;

    /* These should be okay */

    // not a pointer type
    1u8 as u16;
    // cast to less-strictly-aligned type
    (&1u16 as *const u16) as *const u8;
    (&mut 1u16 as *mut u16) as *mut u8;
    // For c_void, we should trust the user. See #2677
    (&1u32 as *const u32 as *const std::os::raw::c_void) as *const u32;
    (&1u32 as *const u32 as *const libc::c_void) as *const u32;
}
