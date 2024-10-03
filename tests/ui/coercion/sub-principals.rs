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
    test::<dyn for<'a> Fn(&'a ()) -> &'a (), dyn Fn(&'static ()) -> &'static ()>();

    trait Foo<'a, 'b> {}
    test::<dyn for<'a, 'b> Foo<'a, 'b>, dyn for<'a> Foo<'a, 'a>>();

    trait Bar<'a> {}
    test::<dyn for<'a> Bar<'a>, dyn Bar<'_>>();
}
