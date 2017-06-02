// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<A> {
    fn foo(&self, a: A) -> A {
        a
    }
}

trait NotRelevant<A> {
    fn nr(&self, a: A) -> A {
        a
    }
}

struct Bar;

impl NotRelevant<usize> for Bar {}

fn main() {
    let f1 = Bar;

    f1.foo(1usize);
    //~^ error: method named `foo` found for type `Bar` in the current scope
    //~| help: items from traits can only be used if the trait is implemented and in scope
    //~| help: candidate #1: `Foo`
}
