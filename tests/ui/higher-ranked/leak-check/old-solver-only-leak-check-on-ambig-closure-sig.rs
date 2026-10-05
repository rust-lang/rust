//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #124440

// A variant of `old-solver-only-leak-check-on-ambig.rs` which
// also relies on us failing to infer the closure signature.
//
// We don't recur into the `?expectation: Foo` bound to get the
// nested higher-ranked `FnMut` bound.
#![allow(warnings)]

trait Foo {}

impl<F> Foo for F where F: FnMut(&()) {}

struct Bar<F> {
    f: F,
}

impl<F> Foo for Bar<F> where F: Foo {}

fn assert_foo<F>(_: F)
where
    Bar<F>: Foo,
{
}

fn main() {
    assert_foo(|_| ());
    //[next]~^ ERROR: the trait bound
}
