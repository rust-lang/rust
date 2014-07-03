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
    fn test6_fn<T: A + Eq>(&self);
    fn test_error7_fn<T: A>(&self);
    fn test_error8_fn<T: B>(&self);
}

impl Foo for int {
    // invalid bound for T, was defined as Eq in trait
    fn test_error1_fn<T: Ord>(&self) {}
    //~^ ERROR in method `test_error1_fn`, type parameter 0 requires bound `core::cmp::Ord`

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error2_fn<T: Eq + B>(&self) {}
    //~^ ERROR in method `test_error2_fn`, type parameter 0 requires bound `B`

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error3_fn<T: B + Eq>(&self) {}
    //~^ ERROR in method `test_error3_fn`, type parameter 0 requires bound `B`

    // multiple bounds, same order as in trait
    fn test3_fn<T: Ord + Eq>(&self) {}

    // multiple bounds, different order as in trait
    fn test4_fn<T: Eq + Ord>(&self) {}

    // parameters in impls must be equal or more general than in the defining trait
    fn test_error5_fn<T: B>(&self) {}
    //~^ ERROR in method `test_error5_fn`, type parameter 0 requires bound `B`

    // bound `std::cmp::Eq` not enforced by this implementation, but this is OK
    fn test6_fn<T: A>(&self) {}

    fn test_error7_fn<T: A + Eq>(&self) {}
    //~^ ERROR in method `test_error7_fn`, type parameter 0 requires bound `core::cmp::Eq`

    fn test_error8_fn<T: C>(&self) {}
    //~^ ERROR in method `test_error8_fn`, type parameter 0 requires bound `C`
}


trait Getter<T> { }

trait Trait {
    fn method<G:Getter<int>>();
}

impl Trait for uint {
    fn method<G: Getter<uint>>() {}
    //~^ ERROR in method `method`, type parameter 0 requires bound `Getter<uint>`
}

fn main() {}

