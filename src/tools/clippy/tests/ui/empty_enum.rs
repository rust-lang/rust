#![allow(dead_code)]
#![warn(clippy::empty_enum)]
// Enable never type to test empty enum lint
#![feature(never_type)]
enum Empty {}
//~^ ERROR: enum with no variants

fn main() {}
