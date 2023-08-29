//@compile-flags: -W clippy::restriction

#![warn(clippy::blanket_clippy_restriction_lints)]

//! Test that the whole restriction group is not enabled
#![warn(clippy::restriction)]
//~^ ERROR: `clippy::restriction` is not meant to be enabled as a group
#![deny(clippy::restriction)]
//~^ ERROR: `clippy::restriction` is not meant to be enabled as a group
#![forbid(clippy::restriction)]
//~^ ERROR: `clippy::restriction` is not meant to be enabled as a group

fn main() {}
