// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(non_snake_case)]
#![allow(dead_code)]

struct Foo;

impl Foo {
    fn Foo_Method() {}
    //~^ ERROR method `Foo_Method` should have a snake case name such as `foo_method`

    // Don't allow two underscores in a row
    fn foo__method(&self) {}
    //~^ ERROR method `foo__method` should have a snake case name such as `foo_method`

    pub fn xyZ(&mut self) {}
    //~^ ERROR method `xyZ` should have a snake case name such as `xy_z`
}

trait X {
    fn ABC();
    //~^ ERROR trait method `ABC` should have a snake case name such as `a_b_c`

    fn a_b_C(&self) {}
    //~^ ERROR trait method `a_b_C` should have a snake case name such as `a_b_c`

    fn something__else(&mut self);
    //~^ ERROR trait method `something__else` should have a snake case name such as `something_else`
}

impl X for Foo {
    // These errors should be caught at the trait definition not the impl
    fn ABC() {}
    fn something__else(&mut self) {}
}

fn Cookie() {}
//~^ ERROR function `Cookie` should have a snake case name such as `cookie`

pub fn bi_S_Cuit() {}
//~^ ERROR function `bi_S_Cuit` should have a snake case name such as `bi_s_cuit`

fn main() { }
