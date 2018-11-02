// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(min_const_unsafe_fn)]

// ok
const unsafe fn foo4() -> i32 { 42 }
const unsafe fn foo5<T>() -> *const T { 0 as *const T }
const unsafe fn foo6<T>() -> *mut T { 0 as *mut T }
const fn no_unsafe() { unsafe {} }

const fn foo8() -> i32 {
    unsafe { foo4() }
}
const fn foo9() -> *const String {
    unsafe { foo5::<String>() }
}
const fn foo10() -> *const Vec<std::cell::Cell<u32>> {
    unsafe { foo6::<Vec<std::cell::Cell<u32>>>() }
}
const unsafe fn foo8_3() -> i32 {
    unsafe { foo4() }
}
const unsafe fn foo9_3() -> *const String {
    unsafe { foo5::<String>() }
}
const unsafe fn foo10_3() -> *const Vec<std::cell::Cell<u32>> {
    unsafe { foo6::<Vec<std::cell::Cell<u32>>>() }
}
// not ok
const unsafe fn foo8_2() -> i32 {
    foo4() //~ ERROR not allowed in const fn
}
const unsafe fn foo9_2() -> *const String {
    foo5::<String>() //~ ERROR not allowed in const fn
}
const unsafe fn foo10_2() -> *const Vec<std::cell::Cell<u32>> {
    foo6::<Vec<std::cell::Cell<u32>>>() //~ ERROR not allowed in const fn
}
const unsafe fn foo30_3(x: *mut usize) -> usize { *x } //~ ERROR not allowed in const fn
//~^ dereferencing raw pointers in constant functions

fn main() {}

const unsafe fn no_union() {
    union Foo { x: (), y: () }
    Foo { x: () }.y //~ ERROR not allowed in const fn
    //~^ unions in const fn
}
