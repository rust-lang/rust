//@aux-build: proc_macros.rs
#![allow(unused)]
#![warn(clippy::let_with_type_underscore)]
#![allow(clippy::let_unit_value)]

extern crate proc_macros;

fn func() -> &'static str {
    ""
}

fn main() {
    // Will lint
    let x: _ = 1;
    let _: _ = 2;
    let x: _ = func();

    let x = 1; // Will not lint, Rust infers this to an integer before Clippy
    let x = func();
    let x: Vec<_> = Vec::<u32>::new();
    let x: [_; 1] = [1];

    // do not lint from procedural macros
    proc_macros::with_span! {
        span
        let x: _ = ();
    };
}
