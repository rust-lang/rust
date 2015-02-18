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
// from traits with static methods.

trait Foo : ::std::marker::MarkerTrait {
    fn foo();
}

fn foo_implicit<T:Foo+'static>(b: Box<T>) -> Box<Foo+'static> {
    b
        //~^ ERROR cannot convert to a trait object
        //~| NOTE method `foo` has no receiver
}

fn foo_explicit<T:Foo+'static>(b: Box<T>) -> Box<Foo+'static> {
    b as Box<Foo>
        //~^ ERROR cannot convert to a trait object
        //~| NOTE method `foo` has no receiver
}

fn main() {
}
