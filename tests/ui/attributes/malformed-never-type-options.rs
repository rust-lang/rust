//! Regression test for #124352
//! The `rustc_*` attribute is malformed, but ICEing without a `feature(rustc_attrs)` is still bad.

#![rustc_never_type_options(: Unsize<U> = "hi")]
//~^ ERROR expected unsuffixed literal, found `:`
//~| ERROR use of an internal attribute

fn main() {}
