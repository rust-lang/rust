#![recursion_limit = foo!()] //~ ERROR malformed `recursion_limit` attribute

macro_rules! foo {
    () => {"128"};
}

fn main() {}
