#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn foo() where for<T> T: Copy {}

fn main() {
    foo();
    //~^ ERROR trait `Copy` is not implemented for `T`
}
