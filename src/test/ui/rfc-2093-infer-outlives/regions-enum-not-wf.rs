// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Various examples of structs whose fields are not well-formed.

#![allow(dead_code)]

trait Dummy<'a> {
  type Out;
}
impl<'a, T> Dummy<'a> for T
where T: 'a
{
  type Out = ();
}
type RequireOutlives<'a, T> = <T as Dummy<'a>>::Out; //~ ERROR the parameter type `T` may not live long enough

enum Ref1<'a, T> {
    Ref1Variant1(RequireOutlives<'a, T>) //~ ERROR the parameter type `T` may not live long enough
}

enum Ref2<'a, T> {
    Ref2Variant1,
    Ref2Variant2(isize, RequireOutlives<'a, T>), //~ ERROR the parameter type `T` may not live long enough
}

enum RefOk<'a, T:'a> {
    RefOkVariant1(&'a T)
}

// This is now well formed. RFC 2093
enum RefIndirect<'a, T> {
    RefIndirectVariant1(isize, RefOk<'a,T>)
}

enum RefDouble<'a, 'b, T> { //~ ERROR 45:1: 48:2: the parameter type `T` may not live long enough [E0309]
    RefDoubleVariant1(&'a RequireOutlives<'b, T>)
        //~^ 46:23: 46:49: the parameter type `T` may not live long enough [E0309]
}

fn main() { }
