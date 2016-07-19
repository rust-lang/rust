// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the parser does not attempt to parse struct literals
// within assignments in if expressions.

#![allow(unused_parens)]

struct Foo {
    foo: usize
}

fn main() {
    let x = 1;
    let y: Foo;

    // `x { ... }` should not be interpreted as a struct literal here
    if x = x {
        //~^ ERROR mismatched types
        //~| expected type `bool`
        //~| found type `()`
        //~| expected bool, found ()
        println!("{}", x);
    }
    // Explicit parentheses on the left should match behavior of above
    if (x = x) {
        //~^ ERROR mismatched types
        //~| expected type `bool`
        //~| found type `()`
        //~| expected bool, found ()
        println!("{}", x);
    }
    // The struct literal interpretation is fine with explicit parentheses on the right
    if y = (Foo { foo: x }) {
        //~^ ERROR mismatched types
        //~| expected type `bool`
        //~| found type `()`
        //~| expected bool, found ()
        println!("{}", x);
    }
}
