#![crate_type = foo!()] //~ ERROR malformed `crate_type` attribute

macro_rules! foo {
    () => {"rlib"};
}

fn main() {}
