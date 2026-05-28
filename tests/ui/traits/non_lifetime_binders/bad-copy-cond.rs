#![feature(non_lifetime_binders)]

fn foo() where for<T> T: Copy {}

fn main() {
    foo();
    //~^ ERROR the trait bound `T: Copy` is not satisfied
}
