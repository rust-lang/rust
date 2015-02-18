// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we normalize associated types that appear in a bound that
// contains a binding. Issue #21664.

#![allow(dead_code)]

use std::marker::MarkerTrait;

pub trait Integral : MarkerTrait {
    type Opposite;
}

impl Integral for i32 {
    type Opposite = u32;
}

impl Integral for u32 {
    type Opposite = i32;
}

pub trait FnLike<A> {
    type R;

    fn dummy(&self, a: A) -> Self::R { loop { } }
}

fn foo<T>()
    where T : FnLike<<i32 as Integral>::Opposite, R=bool>
{
    bar::<T>();
}

fn bar<T>()
    where T : FnLike<u32, R=bool>
{}

fn main() { }
