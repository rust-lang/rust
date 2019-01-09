// aux-build:bang-macro.rs

#![feature(proc_macro_hygiene)]

extern crate bang_macro;
use bang_macro::rewrite;

fn main() {
    assert_eq!(rewrite!("Hello, world!"), "NOT Hello, world!");
}
