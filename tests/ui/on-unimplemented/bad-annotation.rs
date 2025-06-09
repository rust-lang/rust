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
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait EmptyMessage {}

#[rustc_on_unimplemented(lorem(ipsum(dolor)))]
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait Invalid {}

#[rustc_on_unimplemented(message = "x", message = "y")]
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait DuplicateMessage {}

#[rustc_on_unimplemented(message = "x", on(desugared, message = "y"))]
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait OnInWrongPosition {}

#[rustc_on_unimplemented(on(), message = "y")]
//~^ ERROR empty `on`-clause
//~^^ NOTE empty `on`-clause here
trait EmptyOn {}

#[rustc_on_unimplemented(on = "x", message = "y")]
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait ExpectedPredicateInOn {}

#[rustc_on_unimplemented(on(Self = "y"), message = "y")]
trait OnWithoutDirectives {}

#[rustc_on_unimplemented(on(from_desugaring, on(from_desugaring, message = "x")), message = "y")]
//~^ ERROR this attribute must have a value
//~^^ NOTE e.g. `#[rustc_on_unimplemented(message="foo")]`
//~^^^ NOTE expected value here
trait NestedOn {}

#[rustc_on_unimplemented(on("y", message = "y"))]
//~^ ERROR literals inside `on`-clauses are not supported
//~^^ NOTE unexpected literal here
trait UnsupportedLiteral {}

#[rustc_on_unimplemented(on(42, message = "y"))]
//~^ ERROR literals inside `on`-clauses are not supported
//~^^ NOTE unexpected literal here
trait UnsupportedLiteral2 {}

#[rustc_on_unimplemented(on(not(a, b), message = "y"))]
//~^ ERROR expected a single predicate in `not(..)` [E0232]
//~^^ NOTE unexpected quantity of predicates here
trait ExpectedOnePattern {}

#[rustc_on_unimplemented(on(not(), message = "y"))]
//~^ ERROR expected a single predicate in `not(..)` [E0232]
//~^^ NOTE unexpected quantity of predicates here
trait ExpectedOnePattern2 {}

#[rustc_on_unimplemented(on(thing::What, message = "y"))]
//~^ ERROR expected an identifier inside this `on`-clause
//~^^ NOTE expected an identifier here, not `thing::What`
trait KeyMustBeIdentifier {}

#[rustc_on_unimplemented(on(thing::What = "value", message = "y"))]
//~^ ERROR  expected an identifier inside this `on`-clause
//~^^ NOTE expected an identifier here, not `thing::What`
trait KeyMustBeIdentifier2 {}

#[rustc_on_unimplemented(on(aaaaaaaaaaaaaa(a, b), message = "y"))]
//~^ ERROR this predicate is invalid
//~^^ NOTE expected one of `any`, `all` or `not` here, not `aaaaaaaaaaaaaa`
trait InvalidPredicate {}

#[rustc_on_unimplemented(on(something, message = "y"))]
//~^ ERROR invalid flag in `on`-clause
//~^^ NOTE expected one of the `crate_local`, `direct` or `from_desugaring` flags, not `something`
trait InvalidFlag {}

#[rustc_on_unimplemented(on(_Self = "y", message = "y"))]
//~^ ERROR invalid name in `on`-clause
//~^^ NOTE expected one of `cause`, `from_desugaring`, `Self` or any generic parameter of the trait, not `_Self`
trait InvalidName {}

#[rustc_on_unimplemented(on(abc = "y", message = "y"))]
//~^ ERROR invalid name in `on`-clause
//~^^ NOTE expected one of `cause`, `from_desugaring`, `Self` or any generic parameter of the trait, not `abc`
trait InvalidName2 {}
