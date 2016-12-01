// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test implicit coercions involving DSTs and raw pointers.

struct S;
trait T {}
impl T for S {}

struct Foo<T: ?Sized> {
    f: T
}

pub fn main() {
    // Test that we cannot convert from *-ptr to &S and &T
    let x: *const S = &S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &T = x; //~ ERROR mismatched types

    // Test that we cannot convert from *-ptr to &S and &T (mut version)
    let x: *mut S = &mut S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &T = x; //~ ERROR mismatched types

    // Test that we cannot convert an immutable ptr to a mutable one using *-ptrs
    let x: &mut T = &S; //~ ERROR mismatched types
    let x: *mut T = &S; //~ ERROR mismatched types
    let x: *mut S = &S; //~ ERROR mismatched types
}
