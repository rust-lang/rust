//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(negative_bounds)]
#![feature(sized_hierarchy)]

use std::marker::SizeOfVal;

fn foo<T: !SizeOfVal>() {}

fn bar<T: !Sized + SizeOfVal>() {
    foo::<T>();
    //~^ ERROR the trait bound `T: !SizeOfVal` is not satisfied
}

fn main() {
    foo::<()>();
    //~^ ERROR the trait bound `(): !SizeOfVal` is not satisfied
    foo::<str>();
    //~^ ERROR the trait bound `str: !SizeOfVal` is not satisfied
}
