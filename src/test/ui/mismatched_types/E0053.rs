// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn foo(x: u16); //~ NOTE type in trait
    fn bar(&self); //~ NOTE type in trait
}

struct Bar;

impl Foo for Bar {
    fn foo(x: i16) { }
    //~^ ERROR method `foo` has an incompatible type for trait
    //~| NOTE expected u16
    fn bar(&mut self) { }
    //~^ ERROR method `bar` has an incompatible type for trait
    //~| NOTE types differ in mutability
    //~| NOTE expected type `fn(&Bar)`
    //~| NOTE found type `fn(&mut Bar)`
}

fn main() {
}
