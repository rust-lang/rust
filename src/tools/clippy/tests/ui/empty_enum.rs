#![allow(dead_code)]
#![warn(clippy::empty_enum)]
// Enable never type to test empty enum lint
#![feature(never_type)]
enum Empty {}
//~^ empty_enum

fn main() {}
