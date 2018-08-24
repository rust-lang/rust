#![feature(existential_type)]

fn main() {}

// declared but never defined
existential type Bar: std::fmt::Debug; //~ ERROR could not find defining uses
