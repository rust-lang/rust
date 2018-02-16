// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that a default impl still has to have a WF trait ref.

#![feature(specialization)]

trait Foo<'a, T: Eq + 'a> { }

default impl<U> Foo<'static, U> for () {}
//~^ ERROR the trait bound `U: std::cmp::Eq` is not satisfied

fn main(){}
