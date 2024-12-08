#![allow(unknown_or_malformed_diagnostic_attributes)]

trait Foo {}

#[diagnostic::do_not_recommend]
impl<A> Foo for (A,) {}

#[diagnostic::do_not_recommend]
impl<A, B> Foo for (A, B) {}

#[diagnostic::do_not_recommend]
impl<A, B, C> Foo for (A, B, C) {}

impl Foo for i32 {}

fn check(a: impl Foo) {}

fn main() {
    check(());
    //~^ ERROR the trait bound `(): Foo` is not satisfied
}
