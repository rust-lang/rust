//@compile-flags: -W clippy::restriction
//@error-in-other-file: restriction

#![warn(clippy::blanket_clippy_restriction_lints)]

//! Test that the whole restriction group is not enabled
#![warn(clippy::restriction)]
//~^ blanket_clippy_restriction_lints
#![deny(clippy::restriction)]
//~^ blanket_clippy_restriction_lints
#![forbid(clippy::restriction)]
//~^ blanket_clippy_restriction_lints

fn main() {}
