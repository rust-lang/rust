#![recursion_limit = foo!()] //~ ERROR attribute value must be a literal

macro_rules! foo {
    () => {"128"};
}

fn main() {}
