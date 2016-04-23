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

impl Foo<i8> for Bar {}
impl Foo<i16> for Bar {}
impl Foo<i32> for Bar {}

impl Foo<u8> for Bar {}
impl Foo<u16> for Bar {}
impl Foo<u32> for Bar {}

impl NotRelevant<usize> for Bar {}

fn main() {
    let f1 = Bar;

    f1.foo(1usize);
    //~^ error: the trait bound `Bar: Foo<usize>` is not satisfied
    //~| help: the following implementations were found:
    //~| help:   <Bar as Foo<i8>>
    //~| help:   <Bar as Foo<i16>>
    //~| help:   <Bar as Foo<i32>>
    //~| help:   <Bar as Foo<u8>>
    //~| help: and 2 others
}
