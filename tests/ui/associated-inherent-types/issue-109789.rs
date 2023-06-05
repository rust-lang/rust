#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl Foo<fn(&'static ())> {
    type Assoc = u32;
}

trait Other {}
impl Other for u32 {}

// FIXME(inherent_associated_types): Avoid emitting two diagnostics (they only differ in span).
// FIXME(inherent_associated_types): Enhancement: Spruce up the diagnostic by saying something like
// "implementation is not general enough" as is done for traits via
// `try_report_trait_placeholder_mismatch`.

fn bar(_: Foo<for<'a> fn(&'a ())>::Assoc) {}
//~^ ERROR mismatched types
//~| ERROR mismatched types

fn main() {}
