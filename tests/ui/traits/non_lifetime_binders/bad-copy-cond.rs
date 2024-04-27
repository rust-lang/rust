#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn foo() where for<T> T: Copy {}

fn main() {
    foo();
    //~^ ERROR the trait bound `T: Copy` is not satisfied
}
