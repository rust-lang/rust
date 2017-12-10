// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn orange(&self); //~ NOTE previous definition of the value `orange` here
    fn orange(&self); //~ ERROR the name `orange` is defined multiple times
                      //~| NOTE `orange` redefined here
//~| NOTE `orange` must be defined only once in the value namespace of this trait
}

fn main() {}
