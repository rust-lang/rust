// check-pass
// Basic test that show's we can succesfully typeck a `for<T>` where clause.

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Trait {}

impl<T: ?Sized> Trait for T {}

fn foo()
where
    for<T> T: Trait,
{
}

fn main() {
    foo();
}
