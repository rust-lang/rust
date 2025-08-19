//@ check-pass

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait Trait<T: PointeeSized> {}

impl<T: PointeeSized> Trait<T> for i32 {}

fn produce() -> impl for<T> Trait<T> {
    16
}

fn main() {
    let _ = produce();
}
