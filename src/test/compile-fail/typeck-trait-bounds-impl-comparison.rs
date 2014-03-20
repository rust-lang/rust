// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Make sure rustc checks the type parameter bounds in implementations of traits,
// see #2687

trait A {}

trait B: A {}

trait C: A {}

trait Foo {
    fn test_error1_fn<T: Eq>(&self);
    fn test_error2_fn<T: Eq + Ord>(&self);
    fn test_error3_fn<T: Eq + Ord>(&self);
    fn test3_fn<T: Eq + Ord>(&self);
    fn test4_fn<T: Eq + Ord>(&self);
    fn test_error5_fn<T: A>(&self);
    fn test_error6_fn<T: A + Eq>(&self);
    fn test_error7_fn<T: A>(&self);
    fn test_error8_fn<T: B>(&self);
}

impl Foo for int {
    // invalid bound for T, was defined as Eq in trait
    fn test_error1_fn<T: Ord>(&self) {}
    //~^ ERROR bound `std::cmp::Eq` not enforced by this implementation
    //~^^ ERROR implementation bound `std::cmp::Ord` was not specified in trait definition

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error2_fn<T: Eq + B>(&self) {}
    //~^ ERROR bound `std::cmp::Ord` not enforced by this implementation
    //~^^ ERROR implementation bound `B` was not specified in trait definition

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error3_fn<T: B + Eq>(&self) {}
    //~^ ERROR bound `std::cmp::Ord` not enforced by this implementation
    //~^^ ERROR implementation bound `B` was not specified in trait definition

    // multiple bounds, same order as in trait
    fn test3_fn<T: Ord + Eq>(&self) {}

    // multiple bounds, different order as in trait
    fn test4_fn<T: Eq + Ord>(&self) {}

    // parameters in impls must be equal or more general than in the defining trait
    fn test_error5_fn<T: B>(&self) {}
    //~^ ERROR bound `A` not enforced by this implementation
    //~^^ ERROR implementation bound `B` was not specified in trait definition

    fn test_error6_fn<T: A>(&self) {}
    //~^ ERROR bound `std::cmp::Eq` not enforced by this implementation

    fn test_error7_fn<T: A + Eq>(&self) {}
    //~^ ERROR implementation bound `std::cmp::Eq` was not specified in trait definition

    fn test_error8_fn<T: C>(&self) {}
    //~^ ERROR implementation bound `C` was not specified in trait definition
    //~^^ ERROR bound `B` not enforced by this implementation
}


trait Getter<T> { }

trait Trait {
    fn method<G:Getter<int>>();
}

impl Trait for uint {
    fn method<G: Getter<uint>>() {}
    //~^ ERROR requires Getter<uint> but Trait provides Getter<int>
}
fn main() {}
