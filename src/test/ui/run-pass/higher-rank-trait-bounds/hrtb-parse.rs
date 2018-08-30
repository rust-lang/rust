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

// pretty-expanded FIXME #23616

#![allow(unused_variables)]
#![allow(dead_code)]

trait Get<A,R> {
    fn get(&self, arg: A) -> R;
}

// Parse HRTB with explicit `for` in a where-clause:

fn foo00<T>(t: T)
    where T : for<'a> Get<&'a i32, &'a i32>
{
}

fn foo01<T: for<'a> Get<&'a i32, &'a i32>>(t: T)
{
}

// Parse HRTB with explicit `for` in various sorts of types:

fn foo10(t: Box<for<'a> Get<i32, i32>>) { }
fn foo11(t: Box<for<'a> Fn(i32) -> i32>) { }

fn foo20(t: for<'a> fn(i32) -> i32) { }
fn foo21(t: for<'a> unsafe fn(i32) -> i32) { }
fn foo22(t: for<'a> extern "C" fn(i32) -> i32) { }
fn foo23(t: for<'a> unsafe extern "C" fn(i32) -> i32) { }

fn main() {
}
