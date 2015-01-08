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

fn baz<T>()
    where T : for<'x,'y> TheTrait<&'x isize, A = &'y isize>
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
