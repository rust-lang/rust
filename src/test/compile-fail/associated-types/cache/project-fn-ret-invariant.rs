// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]
#![feature(rustc_attrs)]

// Test for projection cache. We should be able to project distinct
// lifetimes from `foo` as we reinstantiate it multiple times, but not
// if we do it just once. In this variant, the region `'a` is used in
// an invariant position, which affects the results.

// revisions: ok oneuse transmute krisskross

#![allow(dead_code, unused_variables)]

use std::marker::PhantomData;

struct Type<'a> {
    // Invariant
    data: PhantomData<fn(&'a u32) -> &'a u32>
}

fn foo<'a>() -> Type<'a> { loop { } }

fn bar<T>(t: T, x: T::Output) -> T::Output
    where T: FnOnce<()>
{
    t()
}

#[cfg(ok)] // two instantiations: OK
fn baz<'a,'b>(x: Type<'a>, y: Type<'b>) -> (Type<'a>, Type<'b>) {
    let a = bar(foo, x);
    let b = bar(foo, y);
    (a, b)
}

// FIXME(#32330)
//#[cfg(oneuse)] // one instantiation: BAD
//fn baz<'a,'b>(x: Type<'a>, y: Type<'b>) -> (Type<'a>, Type<'b>) {
//    let f = foo; // <-- No consistent type can be inferred for `f` here.
//    let a = bar(f, x); //[oneuse] ERROR E0495
//    let b = bar(f, y);
//    (a, b)
//}

// FIXME(#32330)
//#[cfg(transmute)] // one instantiations: BAD
//fn baz<'a,'b>(x: Type<'a>) -> Type<'static> {
//    // Cannot instantiate `foo` with any lifetime other than `'a`,
//    // since it is provided as input.
//
//    bar(foo, x) //[transmute] ERROR E0495
//}

// FIXME(#32330)
//#[cfg(krisskross)] // two instantiations, mixing and matching: BAD
//fn transmute<'a,'b>(x: Type<'a>, y: Type<'b>) -> (Type<'a>, Type<'b>) {
//    let a = bar(foo, y); //[krisskross] ERROR E0495
//    let b = bar(foo, x); //[krisskross] ERROR E0495
//    (a, b)
//}

#[rustc_error]
fn main() { }
//[ok]~^ ERROR compilation successful
//[oneuse]~^^ ERROR compilation successful
//[transmute]~^^^ ERROR compilation successful
//[krisskross]~^^^^ ERROR compilation successful
