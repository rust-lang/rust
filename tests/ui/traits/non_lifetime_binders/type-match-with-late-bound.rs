//@ edition:2021
//@ known-bug: unknown

// Checks that test_type_match code doesn't ICE when predicates have late-bound types

#![feature(non_lifetime_binders)]

async fn walk2<'a, T: 'a>(_: T)
where
    for<F> F: 'a,
{}

fn main() {}
