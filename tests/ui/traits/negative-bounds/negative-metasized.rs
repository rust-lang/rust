//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(negative_bounds)]
#![feature(sized_hierarchy)]

use std::marker::MetaSized;

fn foo<T: !MetaSized>() {}

fn bar<T: !Sized + MetaSized>() {
    foo::<T>();
    //~^ ERROR the trait bound `T: !MetaSized` is not satisfied
}

fn main() {
    foo::<()>();
    //~^ ERROR the trait bound `(): !MetaSized` is not satisfied
    foo::<str>();
    //~^ ERROR the trait bound `str: !MetaSized` is not satisfied
}
