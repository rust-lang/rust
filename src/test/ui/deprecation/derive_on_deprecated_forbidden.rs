// compile-pass

#![forbid(deprecated)]

#[deprecated = "oh no"]
#[derive(Default)]
struct X;

fn main() {}
