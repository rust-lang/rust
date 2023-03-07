// known-bug: unknown
// check-pass

// We currently don't region-check inherent associated type projections at all.

#![feature(inherent_associated_types)]
#![allow(incomplete_features, dead_code)]

struct S<T>(T);

impl S<&'static ()> {
    type T = ();
}

fn usr<'a>() {
    let _: S::<&'a ()>::T; // this should *fail* but it doesn't!
}

fn main() {}
