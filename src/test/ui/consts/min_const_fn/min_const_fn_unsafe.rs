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
const unsafe fn foo4() -> i32 { 42 }
const unsafe fn foo5<T>() -> *const T { 0 as *const T }
const unsafe fn foo6<T>() -> *mut T { 0 as *mut T }
const fn no_unsafe() { unsafe {} }

// not ok
const fn foo8() -> i32 {
    unsafe { foo4() } //~ ERROR calls to `const unsafe fn` in const fns are unstable
}
const fn foo9() -> *const String {
    unsafe { foo5::<String>() } //~ ERROR calls to `const unsafe fn` in const fns are unstable
}
const fn foo10() -> *const Vec<std::cell::Cell<u32>> {
    unsafe { foo6::<Vec<std::cell::Cell<u32>>>() } //~ ERROR calls to `const unsafe fn` in const fns
}
const unsafe fn foo30_3(x: *mut usize) -> usize { *x } //~ ERROR not allowed in const fn
//~^ dereferencing raw pointers in constant functions

fn main() {}

const unsafe fn no_union() {
    union Foo { x: (), y: () }
    Foo { x: () }.y //~ ERROR not allowed in const fn
    //~^ unions in const fn
}
