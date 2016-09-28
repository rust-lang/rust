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
    fn foo(&self, x: u8) -> bool; //~ NOTE trait requires 2 parameters
    fn bar(&self, x: u8, y: u8, z: u8); //~ NOTE trait requires 4 parameters
    fn less(&self); //~ NOTE trait requires 1 parameter
}

struct Bar;

impl Foo for Bar {
    fn foo(&self) -> bool { true } //~ ERROR E0050
                                   //~| NOTE expected 2 parameters, found 1
    fn bar(&self) { } //~ ERROR E0050
                      //~| NOTE expected 4 parameters, found 1
    fn less(&self, x: u8, y: u8, z: u8) { } //~ ERROR E0050
                                            //~| NOTE expected 1 parameter, found 4
}

fn main() {
}
