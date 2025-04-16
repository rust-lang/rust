// ignore-tidy-linelength

#![feature(rustc_attrs)]

#![allow(unused)]

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}` `{Baz}` `{Quux}`"]
trait Foo<Bar, Baz, Quux>
{}

#[rustc_on_unimplemented="a collection of type `{Self}` cannot be built from an iterator over elements of type `{A}`"]
trait MyFromIterator<A> {
    /// Builds a container with elements from an external iterator.
    fn my_from_iter<T: Iterator<Item=A>>(iterator: T) -> Self;
}

#[rustc_on_unimplemented]
//~^ ERROR malformed `rustc_on_unimplemented` attribute
trait BadAnnotation1
{}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{C}>`"]
//~^ ERROR there is no parameter `C` on trait `BadAnnotation2`
trait BadAnnotation2<A,B>
{}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{}>`"]
//~^ ERROR only named generic parameters are allowed
trait BadAnnotation3<A,B>
{}

#[rustc_on_unimplemented(lorem="")]
//~^ ERROR this attribute must have a valid
trait BadAnnotation4 {}

#[rustc_on_unimplemented(lorem(ipsum(dolor)))]
//~^ ERROR this attribute must have a valid
trait BadAnnotation5 {}

#[rustc_on_unimplemented(message="x", message="y")]
//~^ ERROR this attribute must have a valid
trait BadAnnotation6 {}

#[rustc_on_unimplemented(message="x", on(desugared, message="y"))]
//~^ ERROR this attribute must have a valid
trait BadAnnotation7 {}

#[rustc_on_unimplemented(on(), message="y")]
//~^ ERROR empty `on`-clause
trait BadAnnotation8 {}

#[rustc_on_unimplemented(on="x", message="y")]
//~^ ERROR this attribute must have a valid
trait BadAnnotation9 {}

#[rustc_on_unimplemented(on(x="y"), message="y")]
trait BadAnnotation10 {}

#[rustc_on_unimplemented(on(desugared, on(desugared, message="x")), message="y")]
//~^ ERROR this attribute must have a valid
trait BadAnnotation11 {}

pub fn main() {
}
