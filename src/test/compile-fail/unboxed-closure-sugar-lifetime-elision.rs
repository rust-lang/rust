// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the unboxed closure sugar can be used with an arbitrary
// struct type and that it is equivalent to the same syntax using
// angle brackets. This test covers only simple types and in
// particular doesn't test bound regions.

#![feature(unboxed_closures)]
#![allow(dead_code)]

trait Foo<T,U> {
    fn dummy(&self, t: T, u: U);
}

trait Eq<Sized? X> for Sized? { }
impl<Sized? X> Eq<X> for X { }
fn eq<Sized? A,Sized? B:Eq<A>>() { }

fn main() {
    eq::< for<'a> Foo<(&'a int,), &'a int>,
          Foo(&int) -> &int                                   >();
    eq::< for<'a> Foo<(&'a int,), (&'a int, &'a int)>,
          Foo(&int) -> (&int, &int)                           >();

    let _: Foo(&int, &uint) -> &uint; //~ ERROR missing lifetime specifier
}
