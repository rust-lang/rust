// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can parse all the various places that a `for` keyword
// can appear representing universal quantification.

#![feature(unboxed_closures)]
#![allow(unused_variables)]
#![allow(dead_code)]

trait Get<A,R> {
    fn get(&self, arg: A) -> R;
}

// Parse HRTB with explicit `for` in a where-clause:

fn foo00<T>(t: T)
    where T : for<'a> Get<&'a int, &'a int>
{
}

fn foo01<T: for<'a> Get<&'a int, &'a int>>(t: T)
{
}

// Parse HRTB with explicit `for` in various sorts of types:

fn foo10(t: Box<for<'a> Get<int, int>>) { }
fn foo11(t: Box<for<'a> Get(int) -> int>) { }

fn foo20(t: for<'a> fn(int) -> int) { }
fn foo21(t: for<'a> unsafe fn(int) -> int) { }
fn foo22(t: for<'a> extern "C" fn(int) -> int) { }
fn foo23(t: for<'a> unsafe extern "C" fn(int) -> int) { }

fn foo30(t: for<'a> |int| -> int) { }
fn foo31(t: for<'a> unsafe |int| -> int) { }

fn main() {
}
