// https://github.com/rust-lang/rust/issues/107147

#![warn(clippy::needless_pass_by_value)]

struct Foo<'a>(&'a [(); 100]);

fn test(x: Foo<'_>) {}

fn main() {}
