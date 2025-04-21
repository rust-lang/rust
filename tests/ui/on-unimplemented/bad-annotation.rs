#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![allow(unused)]

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}` `{Baz}` `{Quux}`"]
trait Foo<Bar, Baz, Quux> {}

#[rustc_on_unimplemented = "a collection of type `{Self}` cannot \
 be built from an iterator over elements of type `{A}`"]
trait MyFromIterator<A> {
    /// Builds a container with elements from an external iterator.
    fn my_from_iter<T: Iterator<Item = A>>(iterator: T) -> Self;
}

#[rustc_on_unimplemented]
//~^ ERROR malformed `rustc_on_unimplemented` attribute
trait NoContent {}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{C}>`"]
//~^ ERROR cannot find parameter C on this trait
trait ParameterNotPresent<A, B> {}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{}>`"]
//~^ ERROR positional format arguments are not allowed here
trait NoPositionalArgs<A, B> {}

#[rustc_on_unimplemented(lorem = "")]
//~^ ERROR this attribute must have a valid
trait EmptyMessage {}

#[rustc_on_unimplemented(lorem(ipsum(dolor)))]
//~^ ERROR this attribute must have a valid
trait Invalid {}

#[rustc_on_unimplemented(message = "x", message = "y")]
//~^ ERROR this attribute must have a valid
trait DuplicateMessage {}

#[rustc_on_unimplemented(message = "x", on(desugared, message = "y"))]
//~^ ERROR this attribute must have a valid
trait OnInWrongPosition {}

#[rustc_on_unimplemented(on(), message = "y")]
//~^ ERROR empty `on`-clause
trait NoEmptyOn {}

#[rustc_on_unimplemented(on = "x", message = "y")]
//~^ ERROR this attribute must have a valid
trait ExpectedPredicateInOn {}

#[rustc_on_unimplemented(on(x = "y"), message = "y")]
trait OnWithoutDirectives {}

#[rustc_on_unimplemented(on(desugared, on(desugared, message = "x")), message = "y")]
//~^ ERROR this attribute must have a valid
trait NoNestedOn {}

// caught by `OnUnimplementedDirective::parse`, *not* `eval_condition`
#[rustc_on_unimplemented(on("y", message = "y"))]
//~^ ERROR invalid `on`-clause
trait UnsupportedLiteral {}

#[rustc_on_unimplemented(on(not(a, b), message = "y"))]
//~^ ERROR expected 1 cfg-pattern
trait ExpectedOnePattern {}

#[rustc_on_unimplemented(on(thing::What, message = "y"))]
//~^ ERROR `cfg` predicate key must be an identifier
trait KeyMustBeIdentifier {}

#[rustc_on_unimplemented(on(thing::What = "value", message = "y"))]
//~^ ERROR `cfg` predicate key must be an identifier
trait KeyMustBeIdentifier2 {}
