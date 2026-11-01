//
// Make sure rustc checks the type parameter bounds in implementations of traits,
// see #2687

use std::marker;

trait A { }

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

impl Foo for isize {
    // invalid bound for T, was defined as Eq in trait
    fn test_error1_fn<T: Ord>(&self) {}
    //~^ ERROR E0276

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error2_fn<T: Eq + B>(&self) {}
    //~^ ERROR E0276

    // invalid bound for T, was defined as Eq + Ord in trait
    fn test_error3_fn<T: B + Eq>(&self) {}
    //~^ ERROR E0276

    // multiple bounds, same order as in trait
    fn test3_fn<T: Ord + Eq>(&self) {}

    // multiple bounds, different order as in trait
    fn test4_fn<T: Eq + Ord>(&self) {}

    // parameters in impls must be equal or more general than in the defining trait
    fn test_error5_fn<T: B>(&self) {}
    //~^ ERROR E0276

    // bound `std::cmp::Eq` not enforced by this implementation, but this is OK
    fn test6_fn<T: A>(&self) {}

    fn test_error7_fn<T: A + Eq>(&self) {}
    //~^ ERROR E0276

    fn test_error8_fn<T: C>(&self) {}
    //~^ ERROR E0276
}

trait Getter<T> {
    fn get(&self) -> T { loop { } }
}

trait Trait {
    fn method<G:Getter<isize>>(&self);
}

impl Trait for usize {
    fn method<G: Getter<usize>>(&self) {}
    //~^ ERROR E0276
}

fn main() {}
