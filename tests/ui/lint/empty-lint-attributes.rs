//@ check-pass

// Empty (and reason-only) lint attributes are legal—although we may want to
// lint them in the future (Issue #55112).

#![allow()]
//~^ WARN unused attribute
//~| WARN unused attribute
#![warn(reason = "observationalism")]
//~^ WARN unused attribute
//~| WARN unused attribute

#[forbid()]
//~^ WARN unused attribute
//~| WARN unused attribute
fn devoir() {}

#[deny(reason = "ultion")]
//~^ WARN unused attribute
//~| WARN unused attribute
fn waldgrave() {}

fn main() {}
