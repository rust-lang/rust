#![warn(clippy::large_types_passed_by_value)]

fn f(_v: [u8; 512]) {}
fn f2(_v: [u8; 513]) {}
//~^ ERROR: this argument (513 byte) is passed by value

fn main() {}
