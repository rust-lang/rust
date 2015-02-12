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

use std::marker::PhantomFn;

trait Foo<T> {
    type Output;
    fn dummy(&self, t: T, u: Self::Output);
}

trait Eq<X: ?Sized> : PhantomFn<(Self,X)> { }
impl<X: ?Sized> Eq<X> for X { }
fn eq<A: ?Sized,B: ?Sized +Eq<A>>() { }

fn test<'a,'b>() {
    // No errors expected:
    eq::< Foo<(),Output=()>,                       Foo()                         >();
    eq::< Foo<(isize,),Output=()>,                 Foo(isize)                      >();
    eq::< Foo<(isize,usize),Output=()>,            Foo(isize,usize)                 >();
    eq::< Foo<(isize,usize),Output=usize>,         Foo(isize,usize) -> usize         >();
    eq::< Foo<(&'a isize,&'b usize),Output=usize>, Foo(&'a isize,&'b usize) -> usize >();

    // Test that anonymous regions in `()` form are equivalent
    // to fresh bound regions, and that we can intermingle
    // named and anonymous as we choose:
    eq::< for<'x,'y> Foo<(&'x isize,&'y usize),Output=usize>,
          for<'x,'y> Foo(&'x isize,&'y usize) -> usize            >();
    eq::< for<'x,'y> Foo<(&'x isize,&'y usize),Output=usize>,
          for<'x> Foo(&'x isize,&usize) -> usize                  >();
    eq::< for<'x,'y> Foo<(&'x isize,&'y usize),Output=usize>,
          for<'y> Foo(&isize,&'y usize) -> usize                  >();
    eq::< for<'x,'y> Foo<(&'x isize,&'y usize),Output=usize>,
          Foo(&isize,&usize) -> usize                             >();

    // lifetime elision
    eq::< for<'x> Foo<(&'x isize,), Output=&'x isize>,
          Foo(&isize) -> &isize                                   >();

    // Errors expected:
    eq::< Foo<(),Output=()>,
          Foo(char)                                               >();
    //~^^ ERROR not implemented
}

fn main() { }
