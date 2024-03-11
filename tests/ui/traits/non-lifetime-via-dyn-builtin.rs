//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete and may not be safe

fn trivial<A>()
where
    for<B> dyn Fn(A, *const B): Fn(A, *const B),
{
}

fn main() {
    trivial::<u8>();
}
