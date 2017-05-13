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

#![feature(unboxed_closures)]
#![allow(dead_code)]

use std::marker;

trait Foo<'a,T> {
    type Output;
    fn dummy(&'a self) -> &'a (T,Self::Output);
}

trait Eq<X: ?Sized> { fn is_of_eq_type(&self, x: &X) -> bool { true } }
impl<X: ?Sized> Eq<X> for X { }
fn eq<A: ?Sized,B: ?Sized +Eq<A>>() { }

fn same_type<A,B:Eq<A>>(a: A, b: B) { }

fn test<'a,'b>() {
    // Parens are equivalent to omitting default in angle.
    eq::< Foo<(isize,),Output=()>,               Foo(isize)                      >();

    // Here we specify 'static explicitly in angle-bracket version.
    // Parenthesized winds up getting inferred.
    eq::< Foo<'static, (isize,),Output=()>,      Foo(isize)                      >();
}

fn test2(x: &Foo<(isize,),Output=()>, y: &Foo(isize)) {
    // Here, the omitted lifetimes are expanded to distinct things.
    same_type(x, y) //~ ERROR cannot infer
                    //~^ ERROR cannot infer
}

fn main() { }
