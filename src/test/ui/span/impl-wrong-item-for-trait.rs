// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

use std::fmt::Debug;

trait Foo {
    fn bar(&self);
    const MY_CONST: u32;
}

pub struct FooConstForMethod;

impl Foo for FooConstForMethod {
    //~^ ERROR E0046
    //~| NOTE missing `bar` in implementation
    const bar: u64 = 1;
    //~^ ERROR E0323
    //~| NOTE does not match trait
    const MY_CONST: u32 = 1;
}

pub struct FooMethodForConst;

impl Foo for FooMethodForConst {
    //~^ ERROR E0046
    //~| NOTE missing `MY_CONST` in implementation
    fn bar(&self) {}
    fn MY_CONST() {}
    //~^ ERROR E0324
    //~| NOTE does not match trait
}

pub struct FooTypeForMethod;

impl Foo for FooTypeForMethod {
    //~^ ERROR E0046
    //~| NOTE missing `bar` in implementation
    type bar = u64;
    //~^ ERROR E0325
    //~| NOTE does not match trait
    const MY_CONST: u32 = 1;
}

impl Debug for FooTypeForMethod {
}

fn main () {}
