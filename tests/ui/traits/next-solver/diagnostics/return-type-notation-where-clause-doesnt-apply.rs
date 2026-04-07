// Regression test for https://github.com/rust-lang/rust/issues/152887
//@ edition: 2024

#![feature(return_type_notation)]

pub trait Foo {
    async fn bar();
}
trait Bar {}

impl<T: Foo<bar(..): Send>> Foo for T where T: Bar {}
//~^ ERROR not all trait items implemented, missing: `bar`

fn needs_foo(_: impl Foo) {}

trait Mirror {
    type Mirror;
}
impl<T> Mirror for T {
    type Mirror = T;
}

fn hello<T>()
where
    <T as Mirror>::Mirror: Foo,
{
    needs_foo(());
    //~^ ERROR the trait bound `(): Foo` is not satisfied
    //~| ERROR overflow evaluating the requirement
}

fn main() {}
