// aux-build:bang-macro.rs
// ignore-stage1

#![feature(proc_macro_non_items)]

extern crate bang_macro;
use bang_macro::rewrite;

fn main() {
    assert_eq!(rewrite!("Hello, world!"), "NOT Hello, world!");
}
