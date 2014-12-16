// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we correctly prevent users from making trait objects
// from traits with generic methods.

trait Bar {
    fn bar<T>(&self, t: T);
}

fn make_bar<T:Bar>(t: &T) -> &Bar {
    t
        //~^ ERROR `Bar` is not object-safe
        //~| NOTE method `bar` has generic type parameters
}

fn make_bar_explicit<T:Bar>(t: &T) -> &Bar {
    t as &Bar
        //~^ ERROR `Bar` is not object-safe
        //~| NOTE method `bar` has generic type parameters
}

fn main() {
}
