#![allow(unused)]
#![warn(clippy::let_with_type_underscore)]
#![allow(clippy::let_unit_value)]

fn func() -> &'static str {
    ""
}

fn main() {
    // Will lint
    let x: _ = 1;
    let _: _ = 2;
    let x: _ = func();

    let x = 1; // Will not lint, Rust inferres this to an integer before Clippy
    let x = func();
    let x: Vec<_> = Vec::<u32>::new();
    let x: [_; 1] = [1];
}
