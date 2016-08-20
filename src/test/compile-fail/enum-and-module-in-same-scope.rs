// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo { //~ NOTE previous definition of `Foo` here
    X
}

mod Foo { //~ ERROR a type named `Foo` has already been defined
          //~| NOTE already defined
    pub static X: isize = 42;
    fn f() { f() } // Check that this does not result in a resolution error
}

fn main() {}
