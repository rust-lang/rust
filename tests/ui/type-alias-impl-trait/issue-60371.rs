#![feature(type_alias_impl_trait)]

trait Bug {
    type Item: Bug;

    const FUN: fn() -> Self::Item;
}

impl Bug for &() {
    type Item = impl Bug;

    const FUN: fn() -> Self::Item = || ();
    //~^ ERROR the trait bound `(): Bug` is not satisfied
}

fn main() {}
