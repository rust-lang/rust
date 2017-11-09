// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generator_trait)]
#![feature(generators)]

// Test that we cannot create a generator that returns a value of its
// own type.

use std::ops::Generator;

pub fn want_cyclic_generator_return<T>(_: T)
    where T: Generator<Yield = (), Return = T>
{
}

fn supply_cyclic_generator_return() {
    want_cyclic_generator_return(|| {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

pub fn want_cyclic_generator_yield<T>(_: T)
    where T: Generator<Yield = T, Return = ()>
{
}

fn supply_cyclic_generator_yield() {
    want_cyclic_generator_yield(|| {
        //~^ ERROR type mismatch
        if false { yield None.unwrap(); }
        None.unwrap()
    })
}

fn main() { }
