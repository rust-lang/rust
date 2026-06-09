#![deny(unused_attributes)]

#[allow(reason = "I want to allow something")]//~ ERROR unused attribute
#[expect(reason = "I don't know what I'm waiting for")]//~ ERROR unused attribute
#[warn(reason = "This should be warn by default")]//~ ERROR unused attribute
#[deny(reason = "All listed lints are denied")]//~ ERROR unused attribute
#[forbid(reason = "Just some reason")]//~ ERROR unused attribute

#[allow(clippy::box_collection, reason = "This is still valid")]
#[warn(dead_code, reason = "This is also reasonable")]

fn main() {}
