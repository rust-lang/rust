// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait A: Copy {}

struct Foo;

pub trait D {
    fn f<T>(self)
        where T<Bogus = Foo>: A;
        //~^ ERROR associated type bindings are not allowed here [E0229]
        //~| NOTE associate type not allowed here
}

fn main() {}
