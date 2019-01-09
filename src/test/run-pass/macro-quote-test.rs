// Test that a macro can emit delimiters with nothing inside - `()`, `{}`

// aux-build:hello_macro.rs

#![feature(proc_macro_hygiene)]

extern crate hello_macro;

fn main() {
    hello_macro::hello!();
}
