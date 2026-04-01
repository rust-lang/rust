//@ check-pass

#![warn(clippy::empty_enums)]

// `never_type` is not enabled; this test has no stderr file
enum Empty {}

fn main() {}
