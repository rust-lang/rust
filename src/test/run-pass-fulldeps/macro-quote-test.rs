// Test that a macro can emit delimiters with nothing inside - `()`, `{}`

// aux-build:hello_macro.rs
// ignore-stage1

#![feature(proc_macro_non_items, proc_macro_gen)]

extern crate hello_macro;

fn main() {
    hello_macro::hello!();
}
