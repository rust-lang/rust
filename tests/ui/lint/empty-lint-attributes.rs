//@ check-pass

// Empty (and reason-only) lint attributes are legal—although we may want to
// lint them in the future (Issue #55112).

#![allow()]
#![warn(reason = "observationalism")]

#[forbid()]
fn devoir() {}

#[deny(reason = "ultion")]
fn waldgrave() {}

fn main() {}
