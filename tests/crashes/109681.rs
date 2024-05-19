//@ known-bug: #109681

#![crate_type="lib"]
#![feature(linkage)]

#[linkage = "common"]
pub static TEST3: bool = true;

fn main() {}
