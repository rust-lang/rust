//@ revisions: no yes
//@[yes] check-pass

// Issue 110557

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

pub trait Foo {}

#[cfg(no)]
struct Bar<T>(T) where T: Foo;

#[cfg(yes)]
struct Bar<T>(T) where for<H> H: Foo;

impl<T> Drop for Bar<T>
where
    for<H> H: Foo,
//[no]~^ ERROR `Drop` impl requires `H: Foo` but the struct it is implemented for does not
{
    fn drop(&mut self) {}
}

fn main() {}
