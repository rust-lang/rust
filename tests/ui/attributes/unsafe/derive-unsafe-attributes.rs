#![feature(unsafe_attributes)]

#[derive(unsafe(Debug))] //~ ERROR: traits in `#[derive(...)]` don't accept `unsafe(...)`
struct Foo;

fn main() {}
