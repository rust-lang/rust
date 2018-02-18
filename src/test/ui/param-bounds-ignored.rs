// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// must-compile-successfully
#![allow(dead_code, non_camel_case_types)]

use std::rc::Rc;

type SVec<T: Send+Send> = Vec<T>;
//~^ WARN bounds on generic type parameters are ignored in type aliases
type VVec<'b, 'a: 'b+'b> = Vec<&'a i32>;
//~^ WARN bounds on generic lifetime parameters are ignored in type aliases
type WVec<'b, T: 'b+'b> = Vec<T>;
//~^ WARN bounds on generic type parameters are ignored in type aliases
type W2Vec<'b, T> where T: 'b, T: 'b = Vec<T>;
//~^ WARN where clauses are ignored in type aliases

fn foo<'a>(y: &'a i32) {
    // If the bounds above would matter, the code below would be rejected.
    let mut x : SVec<_> = Vec::new();
    x.push(Rc::new(42));

    let mut x : VVec<'static, 'a> = Vec::new();
    x.push(y);

    let mut x : WVec<'static, & 'a i32> = Vec::new();
    x.push(y);

    let mut x : W2Vec<'static, & 'a i32> = Vec::new();
    x.push(y);
}

fn bar1<'a, 'b>(
    x: &'a i32,
    y: &'b i32,
    f: for<'xa, 'xb: 'xa> fn(&'xa i32, &'xb i32) -> &'xa i32)
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked function types
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

fn bar2<'a, 'b, F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>(
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
    x: &'a i32,
    y: &'b i32,
    f: F)
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

fn bar3<'a, 'b, F>(
    x: &'a i32,
    y: &'b i32,
    f: F)
    where F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

fn bar4<'a, 'b, F>(
    x: &'a i32,
    y: &'b i32,
    f: F)
    where for<'xa, 'xb: 'xa> F: Fn(&'xa i32, &'xb i32) -> &'xa i32
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

struct S1<F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>(F);
//~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
struct S2<F>(F) where F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32;
//~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
struct S3<F>(F) where for<'xa, 'xb: 'xa> F: Fn(&'xa i32, &'xb i32) -> &'xa i32;
//~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds

struct S_fnty(for<'xa, 'xb: 'xa> fn(&'xa i32, &'xb i32) -> &'xa i32);
//~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked function types

type T1 = Box<for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>;
//~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds

fn main() {
    let _ : Option<for<'xa, 'xb: 'xa> fn(&'xa i32, &'xb i32) -> &'xa i32> = None;
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked function types
    let _ : Option<Box<for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>> = None;
    //~^ WARN bounds on generic lifetime parameters are ignored in higher-ranked trait bounds
}
