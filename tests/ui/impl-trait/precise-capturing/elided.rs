//@ check-pass

#![feature(precise_capturing)]

fn elided(x: &()) -> impl Sized + use<'_> { x }

fn main() {}
