#![warn(clippy::blanket_clippy_restriction_lints)]

//! Test that the whole restriction group is not enabled
#![warn(clippy::restriction)]
#![deny(clippy::restriction)]
#![forbid(clippy::restriction)]

fn main() {}
