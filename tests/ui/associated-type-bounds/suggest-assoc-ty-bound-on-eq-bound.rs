// Regression test for issue #105056.
//@ edition: 2021

fn f(_: impl Trait<T = Copy>) {}
//~^ ERROR trait objects must include the `dyn` keyword
//~| HELP add `dyn` keyword before this trait
//~| HELP you might have meant to write a bound here
//~| ERROR the trait `Copy` cannot be made into an object

fn g(_: impl Trait<T = std::fmt::Debug + Eq>) {}
//~^ ERROR trait objects must include the `dyn` keyword
//~| HELP add `dyn` keyword before this trait
//~| HELP you might have meant to write a bound here
//~| ERROR only auto traits can be used as additional traits in a trait object
//~| HELP consider creating a new trait
//~| ERROR the trait `Eq` cannot be made into an object

fn h(_: impl Trait<T<> = 'static + for<'a> Fn(&'a ())>) {}
//~^ ERROR trait objects must include the `dyn` keyword
//~| HELP add `dyn` keyword before this trait
//~| HELP you might have meant to write a bound here

// Don't suggest assoc ty bound in trait object types, that's not valid:
type Obj = dyn Trait<T = Clone>;
//~^ ERROR trait objects must include the `dyn` keyword
//~| HELP add `dyn` keyword before this trait

trait Trait { type T; }

fn main() {}
