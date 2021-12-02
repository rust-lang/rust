#![feature(generic_associated_types)]

// check-fail

use std::fmt::Debug;

// We have a `&'a self`, so we need a `Self: 'a`
trait Iterable {
    type Item<'x>;
    //~^ Missing required bounds
    fn iter<'a>(&'a self) -> Self::Item<'a>;
}

/*
impl<T> Iterable for T {
    type Item<'a> = &'a T;
    fn iter<'a>(&'a self) -> Self::Item<'a> {
        self
    }
}
*/

// We have a `&'a T`, so we need a `T: 'x`
trait Deserializer<T> {
    type Out<'x>;
    //~^ Missing required bounds
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a>;
}

/*
impl<T> Deserializer<T> for () {
    type Out<'a> = &'a T;
    fn deserialize<'a>(&self, input: &'a T) -> Self::Out<'a> { input }
}
*/

// We have a `&'b T` and a `'b: 'a`, so it is implied that `T: 'a`. Therefore, we need a `T: 'x`
trait Deserializer2<T> {
    type Out<'x>;
    //~^ Missing required bounds
    fn deserialize2<'a, 'b: 'a>(&self, input1: &'b T) -> Self::Out<'a>;
}

// We have a `&'a T` and a `&'b U`, so we need a `T: 'x` and a `U: 'y`
trait Deserializer3<T, U> {
    type Out<'x, 'y>;
    //~^ Missing required bounds
    fn deserialize2<'a, 'b>(&self, input: &'a T, input2: &'b U) -> Self::Out<'a, 'b>;
}

// `T` is a param on the function, so it can't be named by the associated type
trait Deserializer4 {
    type Out<'x>;
    fn deserialize<'a, T>(&self, input: &'a T) -> Self::Out<'a>;
}

struct Wrap<T>(T);

// We pass `Wrap<T>` and we see `&'z Wrap<T>`, so we require `D: 'x`
trait Des {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, Wrap<T>>;
}
/*
impl Des for () {
    type Out<'x, D> = &'x D; // Not okay
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, Wrap<T>> {
        data
    }
}
*/

// We have `T` and `'z` as GAT substs. Because of `&'z Wrap<T>`, there is an
// implied bound that `T: 'z`, so we require `D: 'x`
trait Des2 {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z Wrap<T>) -> Self::Out<'z, T>;
}
/*
impl Des2 for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a Wrap<T>) -> Self::Out<'a, T> {
        &data.0
    }
}
*/

// We see `&'z T`, so we require `D: 'x`
trait Des3 {
    type Out<'x, D>;
    //~^ Missing required bounds
    fn des<'z, T>(&self, data: &'z T) -> Self::Out<'z, T>;
}
/*
impl Des3 for () {
    type Out<'x, D> = &'x D;
    fn des<'a, T>(&self, data: &'a T) -> Self::Out<'a, T> {
          data
    }
}
*/

// Similar case to before, except with GAT.
trait NoGat<'a> {
    type Bar;
    fn method(&'a self) -> Self::Bar;
}

// Lifetime is not on function; except `Self: 'a`
// FIXME: we require two bounds (`where Self: 'a, Self: 'b`) when we should only require one
trait TraitLifetime<'a> {
    type Bar<'b>;
    //~^ Missing required bounds
    fn method(&'a self) -> Self::Bar<'a>;
}

// Like above, but we have a where clause that can prove what we want
// FIXME: we require two bounds (`where Self: 'a, Self: 'b`) when we should only require one
trait TraitLifetimeWhere<'a> where Self: 'a {
    type Bar<'b>;
    //~^ Missing required bounds
    fn method(&'a self) -> Self::Bar<'a>;
}

// Explicit bound instead of implicit; we want to still error
trait ExplicitBound {
    type Bar<'b>;
    //~^ Missing required bounds
    fn method<'b>(&self, token: &'b ()) -> Self::Bar<'b> where Self: 'b;
}

// The use of the GAT here is not in the return, we don't want to error
trait NotInReturn {
    type Bar<'b>;
    fn method<'b>(&'b self) where Self::Bar<'b>: Debug;
}

// We obviously error for `Iterator`, but we should also error for `Item`
trait IterableTwo {
    type Item<'a>;
    type Iterator<'a>: Iterator<Item = Self::Item<'a>>;
    //~^ Missing required bounds
    fn iter<'a>(&'a self) -> Self::Iterator<'a>;
}

// We also should report region outlives clauses
trait RegionOutlives {
    type Bar<'a, 'b>;
    //~^ Missing required bounds
    fn foo<'x, 'y>(&self, input: &'x &'y ()) -> Self::Bar<'x, 'y>;
}

/*
impl Foo for () {
    type Bar<'a, 'b> = &'a &'b ();
    fn foo<'x, 'y>(&self, input: &'x &'y ()) -> Self::Bar<'x, 'y> {
        input
    }
}
*/

// If there are multiple methods that return the GAT, require a set of clauses
// that can be satisfied by *all* methods
trait MultipleMethods {
    type Bar<'me>;

    fn gimme<'a>(&'a self) -> Self::Bar<'a>;
    fn gimme_default(&self) -> Self::Bar<'static>;
}

fn main() {}
