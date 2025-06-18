//@ check-pass
// Basic test that show's we can successfully typeck a `for<T>` where clause.

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait Trait: PointeeSized {}

impl<T: PointeeSized> Trait for T {}

fn foo()
where
    for<T> T: Trait,
{
}

fn main() {
    foo();
}
