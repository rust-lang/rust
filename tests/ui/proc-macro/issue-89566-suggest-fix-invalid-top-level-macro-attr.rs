//@ run-rustfix

#![derive(Debug)] //~ ERROR `derive` attribute cannot be used at crate level
#[allow(dead_code)]
struct Test {}

fn main() {}
