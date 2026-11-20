// Test that a macro can emit delimiters with nothing inside - `()`, `{}`

//@ run-pass
//@ proc-macro: hello_macro.rs

extern crate hello_macro;

fn main() {
    hello_macro::hello!();
}
