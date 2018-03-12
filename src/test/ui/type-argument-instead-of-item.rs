// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait T<X> {
    type Item;
    type OtherItem;
}

pub struct Foo { i: Box<T<usize, usize, OtherItem=usize>> }
//~^ ERROR wrong number of type arguments: expected 1, found 2
//~| ERROR the value of the associated type `Item` (from the trait `T`) must be specified

fn main() {}
