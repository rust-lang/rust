// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the right implementation is called through a trait
// object when supertraits include multiple references to the
// same trait, with different type parameters.

trait A: PartialEq<Foo> + PartialEq<Bar> { }

struct Foo;
struct Bar;

struct Aimpl;

impl PartialEq<Foo> for Aimpl {
    fn eq(&self, _rhs: &Foo) -> bool {
        true
    }
}

impl PartialEq<Bar> for Aimpl {
    fn eq(&self, _rhs: &Bar) -> bool {
        false
    }
}

impl A for Aimpl { }

fn main() {
    let a = &Aimpl as &A;

    assert!(*a == Foo);
}
