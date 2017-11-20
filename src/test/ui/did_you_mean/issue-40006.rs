// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

impl X { //~ ERROR cannot be made into an object
//~^ ERROR missing
    Y
}

struct S;

trait X { //~ ERROR missing
    X() {}
    fn xxx() { ### } //~ ERROR missing
    //~^ ERROR expected
    L = M; //~ ERROR missing
    Z = { 2 + 3 }; //~ ERROR expected one of
    ::Y (); //~ ERROR expected one of
}

impl S {
    pub hello_method(&self) { //~ ERROR missing
        println!("Hello");
    }
}

fn main() {
    S.hello_method();
}
