//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Verify that the unsize goal can cast a higher-ranked trait goal to
// a non-higer-ranked instantiation.

#![feature(unsize)]

use std::marker::Unsize;

fn test<T: ?Sized, U: ?Sized>()
where
    T: Unsize<U>,
{
}

fn main() {
    test::<dyn for<'a> Fn(&'a ()) -> &'a (), dyn FnOnce(&'static ()) -> &'static ()>();

    trait Foo: for<'a> Bar<'a> {}
    trait Bar<'a> {}
    test::<dyn Foo, dyn Bar<'static>>();
    test::<dyn Foo, dyn Bar<'_>>();
}
