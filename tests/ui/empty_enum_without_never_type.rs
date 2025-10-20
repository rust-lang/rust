//@ check-pass

#![warn(clippy::empty_enum)]

// `never_type` is not enabled; this test has no stderr file
enum Empty {}

fn main() {}
