// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Znll -Zborrowck=mir -Zverbose

// Tests closures that propagate an outlives relationship to their
// creator where the subject is a projection with no regions (`<T as
// Iterator>::Item`, to be exact).

#![allow(warnings)]
#![feature(dyn_trait)]
#![feature(rustc_attrs)]

trait Anything { }

impl<T> Anything for T { }

fn with_signature<'a, T, F>(x: Box<T>, op: F) -> Box<dyn Anything + 'a>
    where F: FnOnce(Box<T>) -> Box<dyn Anything + 'a>
{
    op(x)
}

#[rustc_regions]
fn no_region<'a, T>(x: Box<T>) -> Box<dyn Anything + 'a>
where
    T: Iterator,
{
    with_signature(x, |mut y| Box::new(y.next()))
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR the associated type `<T as std::iter::Iterator>::Item` may not live long enough
}

#[rustc_regions]
fn correct_region<'a, T>(x: Box<T>) -> Box<dyn Anything + 'a>
where
    T: 'a + Iterator,
{
    with_signature(x, |mut y| Box::new(y.next()))
}

#[rustc_regions]
fn wrong_region<'a, 'b, T>(x: Box<T>) -> Box<dyn Anything + 'a>
where
    T: 'b + Iterator,
{
    with_signature(x, |mut y| Box::new(y.next()))
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR the associated type `<T as std::iter::Iterator>::Item` may not live long enough
}

#[rustc_regions]
fn outlives_region<'a, 'b, T>(x: Box<T>) -> Box<dyn Anything + 'a>
where
    T: 'b + Iterator,
    'b: 'a,
{
    with_signature(x, |mut y| Box::new(y.next()))
}

fn main() {}
