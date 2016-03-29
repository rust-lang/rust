// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types in a where clause.

pub trait ToInt {
    fn to_int(&self) -> isize;
}

pub trait GetToInt
{
    type R;

    fn get(&self) -> <Self as GetToInt>::R;
}

fn foo<G>(g: G) -> isize
    where G : GetToInt
{
    ToInt::to_int(&g.get()) //~ ERROR E0277
}

fn bar<G : GetToInt>(g: G) -> isize
    where G::R : ToInt
{
    ToInt::to_int(&g.get()) // OK
}

pub fn main() {
}
