// compile-pass

#![deny(deprecated)]

#[deprecated = "oh no"]
#[derive(Default)]
struct X;

fn main() {}
