// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_attributes)]
#![allow(dead_code, unused_imports)]
#![feature(core, custom_attribute)]

#![foo] //~ ERROR unused attribute

#[foo] //~ ERROR unused attribute
extern crate core;

#[foo] //~ ERROR unused attribute
use std::collections;

#[foo] //~ ERROR unused attribute
extern "C" {
    #[foo] //~ ERROR unused attribute
    fn foo();
}

#[foo] //~ ERROR unused attribute
mod foo {
    #[foo] //~ ERROR unused attribute
    pub enum Foo {
        #[foo] //~ ERROR unused attribute
        Bar,
    }
}

#[foo] //~ ERROR unused attribute
fn bar(f: foo::Foo) {
    match f {
        #[foo] //~ ERROR unused attribute
        foo::Foo::Bar => {}
    }
}

#[foo] //~ ERROR unused attribute
struct Foo {
    #[foo] //~ ERROR unused attribute
    a: isize
}

#[foo] //~ ERROR unused attribute
trait Baz {
    #[foo] //~ ERROR unused attribute
    fn blah(&self);
    #[foo] //~ ERROR unused attribute
    fn blah2(&self) {}
}

fn main() {}
