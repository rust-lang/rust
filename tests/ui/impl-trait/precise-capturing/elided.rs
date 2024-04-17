//@ check-pass

#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn elided(x: &()) -> impl use<'_> Sized { x }

fn main() {}
