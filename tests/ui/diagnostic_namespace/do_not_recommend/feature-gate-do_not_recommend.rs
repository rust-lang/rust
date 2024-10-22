//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(do_not_recommend)]

pub trait Foo {}

impl Foo for i32 {}

pub trait Bar {}

#[diagnostic::do_not_recommend]
impl<T: Foo> Bar for T {}

fn stuff<T: Bar>(_: T) {}

fn main() {
    stuff(1u8);
    //~^ the trait bound `u8: Bar` is not satisfied
}
