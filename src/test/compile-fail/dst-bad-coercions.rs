// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;
trait T {}
impl T for S {}

pub fn main() {
    let x: *const S = &S;
    let y: &S = x; //~ ERROR mismatched types: expected `&S`, found `*const S` (expected &-ptr
    let y: &T = x; //~ ERROR  mismatched types: expected `&T`, found `*const S` (expected &-ptr

    let x: *mut S = &mut S;
    let y: &S = x; //~ ERROR mismatched types: expected `&S`, found `*mut S` (expected &-ptr
    let y: &T = x; //~ ERROR  mismatched types: expected `&T`, found `*mut S` (expected &-ptr

    let x: &mut T = &S; //~ ERROR types differ in mutability
    let x: *mut T = &S; //~ ERROR types differ in mutability
    let x: *mut S = &S;
    //~^ ERROR mismatched types: expected `*mut S`, found `&S` (values differ in mutability)
}