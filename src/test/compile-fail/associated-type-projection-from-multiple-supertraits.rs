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
    //~^ NOTE ambiguous `Color` from `Vehicle`
    //~| NOTE ambiguous `Color` from `Vehicle`
    //~| NOTE ambiguous `Color` from `Vehicle`

    fn go(&self) {  }
}

pub trait Box {
    type Color;
    //~^ NOTE ambiguous `Color` from `Box`
    //~| NOTE ambiguous `Color` from `Box`
    //~| NOTE ambiguous `Color` from `Box`
    //
    fn mail(&self) {  }
}

pub trait BoxCar : Box + Vehicle {
}

fn dent<C:BoxCar>(c: C, color: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
    //~| NOTE ambiguous associated type `Color`
}

fn dent_object<COLOR>(c: BoxCar<Color=COLOR>) {
    //~^ ERROR ambiguous associated type
    //~| ERROR the value of the associated type `Color` (from the trait `Vehicle`) must be specified
    //~| NOTE ambiguous associated type `Color`
    //~| NOTE missing associated type `Color` value
}

fn paint<C:BoxCar>(c: C, d: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
    //~| NOTE ambiguous associated type `Color`
}

pub fn main() { }
