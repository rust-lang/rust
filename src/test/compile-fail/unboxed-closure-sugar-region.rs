// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test interaction between unboxed closure sugar and region
// parameters (should be exactly as if angle brackets were used
// and regions omitted).

#![feature(default_type_params, unboxed_closures)]
#![allow(dead_code)]

use std::kinds::marker;

trait Foo<'a,T,U> {
    fn dummy(&'a self) -> &'a (T,U);
}

trait Eq<Sized? X> for Sized? { }
impl<Sized? X> Eq<X> for X { }
fn eq<Sized? A,Sized? B:Eq<A>>() { }

fn same_type<A,B:Eq<A>>(a: A, b: B) { }

fn test<'a,'b>() {
    // Parens are equivalent to omitting default in angle.
    eq::< Foo<(int,),()>,               Foo(int)                      >();

    // Here we specify 'static explicitly in angle-bracket version.
    // Parenthesized winds up getting inferred.
    eq::< Foo<'static, (int,),()>,      Foo(int)                      >();
}

fn test2(x: &Foo<(int,),()>, y: &Foo(int)) {
    // Here, the omitted lifetimes are expanded to distinct things.
    same_type(x, y) //~ ERROR cannot infer
}

fn main() { }
