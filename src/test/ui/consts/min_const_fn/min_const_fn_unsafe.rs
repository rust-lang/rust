// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-min_const_unsafe_fn

// ok
const unsafe fn ret_i32_no_unsafe() -> i32 { 42 }
const unsafe fn ret_null_ptr_no_unsafe<T>() -> *const T { 0 as *const T }
const unsafe fn ret_null_mut_ptr_no_unsafe<T>() -> *mut T { 0 as *mut T }
const fn no_unsafe() { unsafe {} }

// not ok
const fn call_unsafe_const_fn() -> i32 {
    unsafe { ret_i32_no_unsafe() } //~ ERROR calls to `const unsafe fn` in const fns are unstable
}
const fn call_unsafe_generic_const_fn() -> *const String {
    unsafe { ret_null_ptr_no_unsafe::<String>() }
    //~^ ERROR calls to `const unsafe fn` in const fns are unstable
}
const fn call_unsafe_generic_cell_const_fn() -> *const Vec<std::cell::Cell<u32>> {
    unsafe { ret_null_mut_ptr_no_unsafe::<Vec<std::cell::Cell<u32>>>() }
    //~^ ERROR calls to `const unsafe fn` in const fns
}
const unsafe fn deref_forbidden(x: *mut usize) -> usize { *x }
//~^ dereferencing raw pointers in constant functions

fn main() {}

const unsafe fn no_union() {
    union Foo { x: (), y: () }
    Foo { x: () }.y
    //~^ unions in const fn
}
