//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(sized_hierarchy)]
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

use std::marker::PointeeSized;

trait Id: PointeeSized {
    type Output: PointeeSized;
}

impl<T: PointeeSized> Id for T {
    type Output = T;
}

trait Everyone: PointeeSized {}
impl<T: PointeeSized> Everyone for T {}

fn hello() where for<T> <T as Id>::Output: Everyone {}

fn main() {
    hello();
}
