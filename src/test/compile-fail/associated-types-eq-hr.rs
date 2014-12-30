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

#![feature(associated_types)]

pub trait TheTrait<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

struct IntStruct {
    x: int
}

impl<'a> TheTrait<&'a int> for IntStruct {
    type A = &'a int;

    fn get(&self, t: &'a int) -> &'a int {
        t
    }
}

struct UintStruct {
    x: int
}

impl<'a> TheTrait<&'a int> for UintStruct {
    type A = &'a uint;

    fn get(&self, t: &'a int) -> &'a uint {
        panic!()
    }
}

fn foo<T>()
    where T : for<'x> TheTrait<&'x int, A = &'x int>
{
    // ok for IntStruct, but not UintStruct
}

fn bar<T>()
    where T : for<'x> TheTrait<&'x int, A = &'x uint>
{
    // ok for UintStruct, but not IntStruct
}

fn baz<T>()
    where T : for<'x,'y> TheTrait<&'x int, A = &'y int>
{
    // not ok for either struct, due to the use of two lifetimes
}

pub fn main() {
    foo::<IntStruct>();
    foo::<UintStruct>(); //~ ERROR type mismatch

    bar::<IntStruct>(); //~ ERROR type mismatch
    bar::<UintStruct>();

    baz::<IntStruct>(); //~ ERROR type mismatch
    baz::<UintStruct>(); //~ ERROR type mismatch
}
