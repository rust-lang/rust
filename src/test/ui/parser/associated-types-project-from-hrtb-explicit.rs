// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only -Z continue-parse-after-error

// Test you can't use a higher-ranked trait bound inside of a qualified
// path (just won't parse).

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

fn foo2<I>(x: <I as for<'x> Foo<&'x isize>>::A)
    //~^ ERROR expected identifier, found keyword `for`
    //~| ERROR expected one of `::` or `>`
{
}

pub fn main() {}
