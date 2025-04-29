// This was originally a test for a `ReEmpty` ICE, but became an unintentional test of
// the coinductiveness of WF predicates. That behavior was removed, and thus this is
// also inadvertently a test for the (non-)co-inductiveness of WF predicates.

pub trait Foo {
    type Type;
}

pub struct Bar<T>(<Self as Foo>::Type) where Self: ;
//~^ ERROR the trait bound `Bar<T>: Foo` is not satisfied
//~| ERROR overflow evaluating the requirement `Bar<T> well-formed`

fn main() {}
