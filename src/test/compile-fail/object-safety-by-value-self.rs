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
// from traits with a `fn(self)` method, unless `where Self : Sized`
// is present on the method.

trait Bar {
    fn bar(self);
}

trait Baz {
    fn baz(self: Self);
}

trait Quux {
    // Legal because of the where clause:
    fn baz(self: Self) where Self : Sized;
}

fn make_bar<T:Bar>(t: &T) -> &Bar {
    t
        //~^ ERROR `Bar` is not object-safe
        //~| NOTE method `bar` has a receiver type of `Self`
}

fn make_bar_explicit<T:Bar>(t: &T) -> &Bar {
    t as &Bar
        //~^ ERROR `Bar` is not object-safe
        //~| NOTE method `bar` has a receiver type of `Self`
}

fn make_baz<T:Baz>(t: &T) -> &Baz {
    t
        //~^ ERROR `Baz` is not object-safe
        //~| NOTE method `baz` has a receiver type of `Self`
}

fn make_baz_explicit<T:Baz>(t: &T) -> &Baz {
    t as &Baz
        //~^ ERROR `Baz` is not object-safe
        //~| NOTE method `baz` has a receiver type of `Self`
}

fn make_quux<T:Quux>(t: &T) -> &Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &Quux {
    t as &Quux
}

fn main() {
}
