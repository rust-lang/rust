// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, non_camel_case_types)]

// Test that bounds on higher-kinded lifetime binders are rejected.

fn bar1<'a, 'b>(
    x: &'a i32,
    y: &'b i32,
    f: for<'xa, 'xb: 'xa+'xa> fn(&'xa i32, &'xb i32) -> &'xa i32)
    //~^ ERROR lifetime bounds cannot be used in this context
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

fn bar2<'a, 'b, F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>(
    //~^ ERROR lifetime bounds cannot be used in this context
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
    //~^ ERROR lifetime bounds cannot be used in this context
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
    //~^ ERROR lifetime bounds cannot be used in this context
{
    // If the bound in f's type would matter, the call below would (have to)
    // be rejected.
    f(x, y);
}

struct S1<F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>(F);
//~^ ERROR lifetime bounds cannot be used in this context
struct S2<F>(F) where F: for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32;
//~^ ERROR lifetime bounds cannot be used in this context
struct S3<F>(F) where for<'xa, 'xb: 'xa> F: Fn(&'xa i32, &'xb i32) -> &'xa i32;
//~^ ERROR lifetime bounds cannot be used in this context

struct S_fnty(for<'xa, 'xb: 'xa> fn(&'xa i32, &'xb i32) -> &'xa i32);
//~^ ERROR lifetime bounds cannot be used in this context

type T1 = Box<for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>;
//~^ ERROR lifetime bounds cannot be used in this context

fn main() {
    let _ : Option<for<'xa, 'xb: 'xa> fn(&'xa i32, &'xb i32) -> &'xa i32> = None;
    //~^ ERROR lifetime bounds cannot be used in this context
    let _ : Option<Box<for<'xa, 'xb: 'xa> Fn(&'xa i32, &'xb i32) -> &'xa i32>> = None;
    //~^ ERROR lifetime bounds cannot be used in this context
}
