//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Id {
    type Output: ?Sized;
}

impl<T: ?Sized> Id for T {
    type Output = T;
}

trait Everyone {}
impl<T: ?Sized> Everyone for T {}

fn hello() where for<T> <T as Id>::Output: Everyone {}

fn main() {
    hello();
}
