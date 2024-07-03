#![feature(unsafe_attributes)]

#[derive(unsafe(Debug))] //~ ERROR: traits in `#[derive(...)]` don't accept `unsafe(...)`
struct Foo;

#[unsafe(derive(Debug))] //~ ERROR: is not an unsafe attribute
struct Bar;

fn main() {}
