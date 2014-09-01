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

struct Foo<Sized? T> {
    f: T
}

pub fn main() {
    // Test that we cannot convert from *-ptr to &-ptr
    let x: *const S = &S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &T = x; //~ ERROR mismatched types

    // Test that we cannot convert from *-ptr to &-ptr (mut version)
    let x: *mut S = &mut S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &T = x; //~ ERROR mismatched types

    // Test that we cannot convert an immutable ptr to a mutable one using *-ptrs
    let x: &mut T = &S; //~ ERROR types differ in mutability
    let x: *mut T = &S; //~ ERROR types differ in mutability
    let x: *mut S = &S;
    //~^ ERROR mismatched types

    // The below four sets of tests test that we cannot implicitly deref a *-ptr
    // during a coercion.
    let x: *const S = &S;
    let y: *const T = x;  //~ ERROR mismatched types

    let x: *mut S = &mut S;
    let y: *mut T = x;  //~ ERROR mismatched types

    let x: *const Foo<S> = &Foo {f: S};
    let y: *const Foo<T> = x;  //~ ERROR mismatched types

    let x: *mut Foo<S> = &mut Foo {f: S};
    let y: *mut Foo<T> = x;  //~ ERROR mismatched types
}
