//@ edition: 2024
//@ compile-flags: -Zforce-unstable-if-unmarked

// Auxiliary crate that uses `-Zforce-unstable-if-unmarked` to export an
// "unstable" trait.

pub trait ForeignTrait {}
