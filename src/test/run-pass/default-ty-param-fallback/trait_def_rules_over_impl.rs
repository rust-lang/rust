// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

trait Bar {
    // Current we consider only the default in the trait,
    // never the one in the impl.
    // Is this good? Is it bad? Is it just the way it works?
    // If this is ok then we should forbid writing the default in the impl.
    fn method<A:Default=String>(&self) -> A;
}

struct Foo;

impl Bar for Foo {
    fn method<A:Default>(&self) -> A {
        A::default()
    }
}

fn main() {
    Foo.method();
}
