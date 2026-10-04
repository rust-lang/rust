//@ run-pass
//@ proc-macro: bang-macro.rs
//@ ignore-backends: gcc

extern crate bang_macro;
use bang_macro::rewrite;

fn main() {
    assert_eq!(rewrite!("Hello, world!"), "NOT Hello, world!");
}
