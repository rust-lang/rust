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

#![feature(default_type_params)]
#![allow(dead_code)]

struct Foo<T,U,V=T> {
    t: T, u: U
}

trait Eq<X> { }
impl<X> Eq<X> for X { }
fn eq<A,B:Eq<A>>() { }

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
