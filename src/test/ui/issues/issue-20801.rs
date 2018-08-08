// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We used to ICE when moving out of a `*mut T` or `*const T`.

struct T(u8);

static mut GLOBAL_MUT_T: T = T(0);

static GLOBAL_T: T = T(0);

fn imm_ref() -> &'static T {
    unsafe { &GLOBAL_T }
}

fn mut_ref() -> &'static mut T {
    unsafe { &mut GLOBAL_MUT_T }
}

fn mut_ptr() -> *mut T {
    unsafe { 0 as *mut T }
}

fn const_ptr() -> *const T {
    unsafe { 0 as *const T }
}

pub fn main() {
    let a = unsafe { *mut_ref() };
    //~^ ERROR cannot move out of borrowed content

    let b = unsafe { *imm_ref() };
    //~^ ERROR cannot move out of borrowed content

    let c = unsafe { *mut_ptr() };
    //~^ ERROR cannot move out of dereference of raw pointer

    let d = unsafe { *const_ptr() };
    //~^ ERROR cannot move out of dereference of raw pointer
}
