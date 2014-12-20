// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a where clause attached to a method allows us to add
// additional constraints to a parameter out of scope.

struct Foo<T> {
    value: T
}

struct Bar; // does not implement Eq

impl<T> Foo<T> {
    fn equals(&self, u: &Foo<T>) -> bool where T : Eq {
        self.value == u.value
    }
}

fn main() {
    let x = Foo { value: Bar };
    x.equals(&x);
    //~^ ERROR the trait `core::cmp::Eq` is not not implemented
}
