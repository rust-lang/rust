//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Make sure that having an applicable user-written
// and builtin impl is ambiguous.

trait Equals<T> {}

impl<T> Equals<T> for T {}

fn impls_equals<T: Equals<U>, U>() {}

fn main() {
    impls_equals::<dyn Equals<u32>, _>();
    //~^ ERROR type annotations needed
}
