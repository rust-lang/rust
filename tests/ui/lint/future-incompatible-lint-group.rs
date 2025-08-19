//! Test that future_incompatible lint group only includes edition-independent lints

// Ensure that the future_incompatible lint group only includes
// lints for changes that are not tied to an edition
#![deny(future_incompatible)]

enum E { V }

trait Tr1 {
    type V;
    fn foo() -> Self::V;
}

impl Tr1 for E {
    type V = u8;

    // Error since this is a `future_incompatible` lint
    fn foo() -> Self::V { 0 }
    //~^ ERROR ambiguous associated item
    //~| WARN this was previously accepted
}

trait Tr2 {
    // Warn only since this is not a `future_incompatible` lint
    fn f(u8) {}
    //~^ WARN anonymous parameters are deprecated
    //~| WARN this is accepted in the current edition
}

fn main() {}
