// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check testing of equality constraints in a higher-ranked context.

pub trait TheTrait<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

struct IntStruct {
    x: isize
}

impl<'a> TheTrait<&'a isize> for IntStruct {
    type A = &'a isize;

    fn get(&self, t: &'a isize) -> &'a isize {
        t
    }
}

struct UintStruct {
    x: isize
}

impl<'a> TheTrait<&'a isize> for UintStruct {
    type A = &'a usize;

    fn get(&self, t: &'a isize) -> &'a usize {
        panic!()
    }
}

struct Tuple {
}

impl<'a> TheTrait<(&'a isize, &'a isize)> for Tuple {
    type A = &'a isize;

    fn get(&self, t: (&'a isize, &'a isize)) -> &'a isize {
        t.0
    }
}

fn foo<T>()
    where T : for<'x> TheTrait<&'x isize, A = &'x isize>
{
    // ok for IntStruct, but not UintStruct
}

fn bar<T>()
    where T : for<'x> TheTrait<&'x isize, A = &'x usize>
{
    // ok for UintStruct, but not IntStruct
}

fn tuple_one<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize), A = &'x isize>
{
    // not ok for tuple, two lifetimes and we pick first
}

fn tuple_two<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize), A = &'y isize>
{
    // not ok for tuple, two lifetimes and we pick second
}

fn tuple_three<T>()
    where T : for<'x> TheTrait<(&'x isize, &'x isize), A = &'x isize>
{
    // ok for tuple
}

fn tuple_four<T>()
    where T : for<'x,'y> TheTrait<(&'x isize, &'y isize)>
{
    // not ok for tuple, two lifetimes, and lifetime matching is invariant
}

pub fn main() {
    foo::<IntStruct>();
    foo::<UintStruct>(); //~ ERROR type mismatch

    bar::<IntStruct>(); //~ ERROR type mismatch
    bar::<UintStruct>();

    tuple_one::<Tuple>();
    //~^ ERROR E0277
    //~| ERROR type mismatch

    tuple_two::<Tuple>();
    //~^ ERROR E0277
    //~| ERROR type mismatch

    tuple_three::<Tuple>();

    tuple_four::<Tuple>();
    //~^ ERROR E0277
}
