//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(do_not_recommend)]

trait Foo {}

#[do_not_recommend]
impl<T> Foo for T where T: Send {}
//[current]~^ NOTE required for `*mut ()` to implement `Foo`
//[current]~| NOTE unsatisfied trait bound introduced here

fn needs_foo<T: Foo>() {}
//~^ NOTE required by a bound in `needs_foo`
//~| NOTE required by this bound in `needs_foo`

fn main() {
    needs_foo::<*mut ()>();
    //~^ ERROR the trait bound `*mut (): Foo` is not satisfied
    //[current]~| NOTE the trait `Send` is not implemented for `*mut ()`
    //[next]~| NOTE the trait `Foo` is not implemented for `*mut ()`
}
