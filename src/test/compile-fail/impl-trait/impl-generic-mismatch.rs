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
    fn foo(&self, _: &impl Debug);
}

impl Foo for () {
    fn foo<U: Debug>(&self, _: &U) { }
    //~^ Error method `foo` has incompatible signature for trait
}

trait Bar {
    fn bar<U: Debug>(&self, _: &U);
}

impl Bar for () {
    fn bar(&self, _: &impl Debug) { }
    //~^ Error method `bar` has incompatible signature for trait
}

// With non-local trait (#49841):

use std::hash::{Hash, Hasher};

struct X;

impl Hash for X {
    fn hash(&self, hasher: &mut impl Hasher) {}
    //~^ Error method `hash` has incompatible signature for trait
}

fn main() {}
