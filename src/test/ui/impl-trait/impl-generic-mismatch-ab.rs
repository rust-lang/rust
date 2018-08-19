// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Debug;

trait Foo {
    fn foo<A: Debug>(&self, a: &A, b: &impl Debug);
}

impl Foo for () {
    fn foo<B: Debug>(&self, a: &impl Debug, b: &B) { }
    //~^ ERROR method `foo` has an incompatible type for trait
}

fn main() {}
