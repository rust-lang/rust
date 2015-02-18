// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we get an error when you use `<Self as Get>::Value` in
// the trait definition but `Self` does not, in fact, implement `Get`.

trait Get : ::std::marker::MarkerTrait {
    type Value;
}

trait Other {
    fn uhoh<U:Get>(&self, foo: U, bar: <Self as Get>::Value) {}
    //~^ ERROR the trait `Get` is not implemented for the type `Self`
}

impl<T:Get> Other for T {
    fn uhoh<U:Get>(&self, foo: U, bar: <(T, U) as Get>::Value) {}
    //~^ ERROR the trait `Get` is not implemented for the type `(T, U)`
    //~| ERROR the trait `Get` is not implemented for the type `(T, U)`
}

fn main() { }
