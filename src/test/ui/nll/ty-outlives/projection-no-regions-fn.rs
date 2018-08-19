// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Zborrowck=mir -Zverbose

#![allow(warnings)]

trait Anything { }

impl<T> Anything for T { }

fn no_region<'a, T>(mut x: T) -> Box<dyn Anything + 'a>
where
    T: Iterator,
{
    Box::new(x.next())
    //~^ WARNING not reporting region error due to nll
    //~| the associated type `<T as std::iter::Iterator>::Item` may not live long enough
}

fn correct_region<'a, T>(mut x: T) -> Box<dyn Anything + 'a>
where
    T: 'a + Iterator,
{
    Box::new(x.next())
}

fn wrong_region<'a, 'b, T>(mut x: T) -> Box<dyn Anything + 'a>
where
    T: 'b + Iterator,
{
    Box::new(x.next())
    //~^ WARNING not reporting region error due to nll
    //~| the associated type `<T as std::iter::Iterator>::Item` may not live long enough
}

fn outlives_region<'a, 'b, T>(mut x: T) -> Box<dyn Anything + 'a>
where
    T: 'b + Iterator,
    'b: 'a,
{
    Box::new(x.next())
}

fn main() {}
