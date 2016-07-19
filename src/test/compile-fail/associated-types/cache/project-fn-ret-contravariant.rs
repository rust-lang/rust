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
// an contravariant position, which affects the results.

// revisions: ok oneuse transmute krisskross

#![allow(dead_code, unused_variables)]

fn foo<'a>() -> &'a u32 { loop { } }

fn bar<T>(t: T, x: T::Output) -> T::Output
    where T: FnOnce<()>
{
    t()
}

#[cfg(ok)] // two instantiations: OK
fn baz<'a,'b>(x: &'a u32, y: &'b u32) -> (&'a u32, &'b u32) {
    let a = bar(foo, x);
    let b = bar(foo, y);
    (a, b)
}

#[cfg(oneuse)] // one instantiation: OK (surprisingly)
fn baz<'a,'b>(x: &'a u32, y: &'b u32) -> (&'a u32, &'b u32) {
    let f /* : fn() -> &'static u32 */ = foo; // <-- inferred type annotated
    let a = bar(f, x); // this is considered ok because fn args are contravariant...
    let b = bar(f, y); // ...and hence we infer T to distinct values in each call.
    (a, b)
}

// FIXME(#32330)
//#[cfg(transmute)] // one instantiations: BAD
//fn baz<'a,'b>(x: &'a u32) -> &'static u32 {
//    bar(foo, x) //[transmute] ERROR E0495
//}

// FIXME(#32330)
//#[cfg(krisskross)] // two instantiations, mixing and matching: BAD
//fn transmute<'a,'b>(x: &'a u32, y: &'b u32) -> (&'a u32, &'b u32) {
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
