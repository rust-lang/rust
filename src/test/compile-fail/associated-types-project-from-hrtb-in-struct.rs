// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check projection of an associated type out of a higher-ranked trait-bound
// in the context of a struct definition.

#![feature(associated_types)]

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

struct SomeStruct<I : for<'x> Foo<&'x int>> {
    field: I::A
    //~^ ERROR cannot extract an associated type from a higher-ranked trait bound in this context
}

struct AnotherStruct<I : for<'x> Foo<&'x int>> {
    field: <I as Foo<&int>>::A
    //~^ ERROR missing lifetime specifier
}

struct YetAnotherStruct<'a, I : for<'x> Foo<&'x int>> {
    field: <I as Foo<&'a int>>::A
}

pub fn main() {}
