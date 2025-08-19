#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait Other<U: PointeeSized>: PointeeSized {}

impl<U: PointeeSized> Other<U> for U {}

#[rustfmt::skip]
fn foo<U: PointeeSized>()
where
    for<T> T: Other<U> {}

fn bar() {
    foo::<_>();
    //~^ ERROR the trait bound `T: Other<_>` is not satisfied
}

fn main() {}
