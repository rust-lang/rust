// Regression test for issue #105056.
//@ edition: 2021

fn f(_: impl Trait<T = Copy>) {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP you might have meant to write a bound here

fn g(_: impl Trait<T = std::fmt::Debug + Eq>) {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP you might have meant to write a bound here

fn h(_: impl Trait<T<> = 'static + for<'a> Fn(&'a ())>) {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP you might have meant to write a bound here

// Don't suggest assoc ty bound in trait object types, that's not valid:
type Obj = dyn Trait<T = Clone>;
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object

trait Trait { type T; }

fn main() {}
