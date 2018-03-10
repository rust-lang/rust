// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test ignored_generic_bounds lint warning about bounds in type aliases

// must-compile-successfully
#![allow(dead_code)]

use std::rc::Rc;

type SVec<T: Send+Send> = Vec<T>;
//~^ WARN bounds on generic parameters are ignored in type aliases
type S2Vec<T> where T: Send = Vec<T>;
//~^ WARN where clauses are ignored in type aliases
type VVec<'b, 'a: 'b+'b> = (&'b u32, Vec<&'a i32>);
//~^ WARN bounds on generic parameters are ignored in type aliases
type WVec<'b, T: 'b+'b> = (&'b u32, Vec<T>);
//~^ WARN bounds on generic parameters are ignored in type aliases
type W2Vec<'b, T> where T: 'b, T: 'b = (&'b u32, Vec<T>);
//~^ WARN where clauses are ignored in type aliases

static STATIC : u32 = 0;

fn foo<'a>(y: &'a i32) {
    // If any of the bounds above would matter, the code below would be rejected.
    // This can be seen when replacing the type aliases above by newtype structs.
    // (The type aliases have no unused parameters to make that a valid transformation.)
    let mut x : SVec<_> = Vec::new();
    x.push(Rc::new(42)); // is not send

    let mut x : S2Vec<_> = Vec::new();
    x.push(Rc::new(42)); // is not send

    let mut x : VVec<'static, 'a> = (&STATIC, Vec::new());
    x.1.push(y); // 'a: 'static does not hold

    let mut x : WVec<'static, &'a i32> = (&STATIC, Vec::new());
    x.1.push(y); // &'a i32: 'static does not hold

    let mut x : W2Vec<'static, &'a i32> = (&STATIC, Vec::new());
    x.1.push(y); // &'a i32: 'static does not hold
}

fn main() {}
