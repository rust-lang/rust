// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints in a where clause where the type being
// equated appears in a supertrait.

pub trait Vehicle {
    type Color;

    fn go(&self) {  }
}

pub trait Box {
    type Color;

    fn mail(&self) {  }
}

fn a<C:Vehicle+Box>(_: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

fn b<C>(_: C::Color) where C : Vehicle+Box {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

fn c<C>(_: C::Color) where C : Vehicle, C : Box {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

struct D<X>;
impl<X> D<X> where X : Vehicle {
    fn d(&self, _: X::Color) where X : Box { }
    //~^ ERROR ambiguous associated type `Color` in bounds of `X`
}

trait E<X:Vehicle> {
    fn e(&self, _: X::Color) where X : Box;
    //~^ ERROR ambiguous associated type `Color` in bounds of `X`

    fn f(&self, _: X::Color) where X : Box { }
    //~^ ERROR ambiguous associated type `Color` in bounds of `X`
}

pub fn main() { }
