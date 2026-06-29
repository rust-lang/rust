//@ compile-flags: -Zinternal-testing-features
#![feature(negative_bounds)]

fn foo<T: !Sized>() {}

fn main() {
    foo::<()>();
    //~^ ERROR the trait bound `(): !Sized` is not satisfied
}
