// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test interaction between unboxed closure sugar and default type
// parameters (should be exactly as if angle brackets were used).

#![feature(default_type_params, unboxed_closures)]
#![allow(dead_code)]

trait Foo<T,U,V=T> {
    fn dummy(&self, t: T, u: U, v: V);
}

trait Eq<Sized? X> for Sized? { }
impl<Sized? X> Eq<X> for X { }
fn eq<Sized? A,Sized? B>() where A : Eq<B> { }

fn test<'a,'b>() {
    // Parens are equivalent to omitting default in angle.
    eq::< Foo<(int,),()>,               Foo(int)                      >();

    // In angle version, we supply something other than the default
    eq::< Foo<(int,),(),int>,           Foo(int)                      >();
    //~^ ERROR not implemented

    // Supply default explicitly.
    eq::< Foo<(int,),(),(int,)>,        Foo(int)                      >();
}

fn main() { }
