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
// in the context of a function signature.

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

fn foo2<I : for<'x> Foo<&'x isize>>(
    x: I::A)
    //~^ ERROR cannot extract an associated type from a higher-ranked trait bound in this context
{
    // This case is illegal because we have to instantiate `'x`, and
    // we don't know what region to instantiate it with.
    //
    // This could perhaps be made equivalent to the examples below,
    // specifically for fn signatures.
}

fn foo3<I : for<'x> Foo<&'x isize>>(
    x: <I as Foo<&isize>>::A)
{
    // OK, in this case we spelled out the precise regions involved, though we left one of
    // them anonymous.
}

fn foo4<'a, I : for<'x> Foo<&'x isize>>(
    x: <I as Foo<&'a isize>>::A)
{
    // OK, in this case we spelled out the precise regions involved.
}


pub fn main() {}
