//@ check-pass

#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn elided(x: &()) -> impl Sized + use<'_> { x }

fn main() {}
