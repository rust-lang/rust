//@ compile-flags: -Z threads=2

const FOO: usize = FOO; //~ERROR cycle detected when simplifying constant for the type system `FOO`

fn main() {}
