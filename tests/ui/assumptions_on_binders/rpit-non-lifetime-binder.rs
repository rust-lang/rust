//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

use std::marker::PointeeSized;

trait Trait<T: PointeeSized> {}

impl<T: PointeeSized> Trait<T> for i32 {}

fn produce() -> impl for<T> Trait<T> {
    16
}

fn main() {
    let _ = produce();
}
