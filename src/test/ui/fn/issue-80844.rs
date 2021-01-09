#![feature(generators)]

// Functions with a type placeholder `_` as the return type should
// not suggest returning the unnameable type of generators.
// This is a regression test of #80844

fn returns_generator() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures [E0121]
//~| NOTE not allowed in type signatures
//~| HELP consider using a `Generator` trait bound
//~| NOTE for more information on generators
    || yield 0
}

fn main() {}
