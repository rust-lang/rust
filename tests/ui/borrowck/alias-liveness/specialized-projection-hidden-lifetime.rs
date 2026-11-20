//@ edition: 2024
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Znext-solver=globally -Zassumptions-on-binders -Zpolonius=off
//@ [polonius] compile-flags: -Znext-solver=globally -Zassumptions-on-binders -Zpolonius=next

// This is similar in spirit to `projection-higher-ranked-bounds.rs`, but throws
// in the added wrinkle that the *base* impl does not contain the outlives bound,
// but a specialized impl does.

#![feature(specialization)]
#![allow(incomplete_features)]
#![forbid(unsafe_code)]

use std::fmt::Debug;

trait Tr<'a, 'b, 'c> {
    type A: Debug + 'a + 'b;
    fn mk(c: &'c u8) -> Self::A;
}

impl<'a, 'b, 'c, T> Tr<'a, 'b, 'c> for T {
    default type A = ();
    default fn mk(_: &'c u8) -> Self::A {
        loop {}
    }
}

struct Special;

impl<'a, 'b, 'c: 'a + 'b> Tr<'a, 'b, 'c> for Special {
    type A = &'c u8;
    fn mk(c: &'c u8) -> Self::A {
        c
    }
}

fn run<'a, 'b, 'c, T: Tr<'a, 'b, 'c>>(c: &'c u8) -> T::A {
    T::mk(c)
}

fn generic<T: for<'a, 'b, 'c> Tr<'a, 'b, 'c>>() {
    let escaped;
    {
        let v = vec![42];
        escaped = run::<T>(&v[0]);
        //~^ ERROR `v` does not live long enough
    }
    let reused = vec![7];
    println!("escaped = {escaped:?}, reused = {reused:?}");
}

fn main() {
    generic::<Special>();
}
