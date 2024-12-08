//@ run-pass
//@ proc-macro: bang-macro.rs

extern crate bang_macro;
use bang_macro::rewrite;

fn main() {
    assert_eq!(rewrite!("Hello, world!"), "NOT Hello, world!");
}
