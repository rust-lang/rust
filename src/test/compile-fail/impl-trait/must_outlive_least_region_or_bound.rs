// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;

fn elided(x: &i32) -> impl Copy { x }
//~^ ERROR explicit lifetime required in the type of `x` [E0621]

fn explicit<'a>(x: &'a i32) -> impl Copy { x }
//~^ ERROR cannot infer an appropriate lifetime

trait LifetimeTrait<'a> {}
impl<'a> LifetimeTrait<'a> for &'a i32 {}

fn with_bound<'a>(x: &'a i32) -> impl LifetimeTrait<'a> + 'static { x }
//~^ ERROR cannot infer an appropriate lifetime

// Tests that a closure type contianing 'b cannot be returned from a type where
// only 'a was expected.
fn move_lifetime_into_fn<'a, 'b>(x: &'a u32, y: &'b u32) -> impl Fn(&'a u32) {
    //~^ ERROR lifetime mismatch
    move |_| println!("{}", y)
}

fn ty_param_wont_outlive_static<T:Debug>(x: T) -> impl Debug + 'static {
    //~^ ERROR the parameter type `T` may not live long enough
    x
}

fn main() {}
