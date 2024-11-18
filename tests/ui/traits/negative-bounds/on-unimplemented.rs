//@ reference: attributes.diagnostic.on_unimplemented.intro

#![feature(negative_bounds)]

#[diagnostic::on_unimplemented(message = "this ain't fooing")]
trait Foo {}
struct NotFoo;

fn hello() -> impl !Foo {
    //~^ ERROR the trait bound `NotFoo: !Foo` is not satisfied
    NotFoo
}

fn main() {}
