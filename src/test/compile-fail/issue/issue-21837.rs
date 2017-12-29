// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Bound {}
pub struct Foo<T: Bound>(T);

pub trait Trait1 {}
impl<T: Bound> Trait1 for Foo<T> {}

pub trait Trait2 {}
impl<T> Trait2 for Foo<T> {} //~ ERROR the trait bound `T: Bound` is not satisfied

fn main() {}
