// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z continue-parse-after-error

// Test that we can recover from missing braces in the parser.

trait Foo {
    fn bar() {
        let x = foo(); //~ ERROR cannot find function `foo` in this scope

}

fn main() {
    let x = y.;  //~ ERROR unexpected token
                 //~^ ERROR cannot find value `y` in this scope
} //~ ERROR this file contains an un-closed delimiter
