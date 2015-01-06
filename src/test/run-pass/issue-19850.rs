// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `<Type as Trait>::Output` and `Self::Output` are accepted as type annotations in let
// bindings

trait Int {
    fn one() -> Self;
    fn leading_zeros(self) -> uint;
}

trait Foo {
    type T : Int;

    fn test(&self) {
        let r: <Self as Foo>::T = Int::one();
        let r: Self::T = Int::one();
    }
}

fn main() {}
