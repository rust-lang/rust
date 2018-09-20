// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(type_alias_missing_bounds)]

pub trait Trait {}

pub struct Foo<T: Trait>(T);

pub struct Qux;

pub type Bar<T = Qux> = Foo<T>;
//~^ ERROR the trait bound `T: Trait` is not satisfied

pub type Baz<T: Trait = Qux> = Foo<T>;
//~^ ERROR the trait bound `Qux: Trait` is not satisfied

fn main() {}
