// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test method calls with self as an argument cannot subvert type checking.

struct Foo;

impl Foo {
    fn bar(&self) {}
}

fn main() {
    let x = Foo;
    Foo::bar(x); //~  ERROR mismatched types
                 //~| expected type `&Foo`
                 //~| found type `Foo`
                 //~| expected &Foo, found struct `Foo`
    Foo::bar(&42); //~  ERROR mismatched types
                      //~| expected type `&Foo`
                      //~| found type `&{integer}`
                      //~| expected struct `Foo`, found integral variable
}
