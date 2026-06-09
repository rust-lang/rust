//@ run-pass
//@ proc-macro: basic.rs
//@ ignore-backends: gcc

extern crate basic;

fn main() {
    basic::run_tests!();
}
