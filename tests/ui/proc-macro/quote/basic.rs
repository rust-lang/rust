//@ run-pass
//@ proc-macro: basic.rs

extern crate basic;

fn main() {
    basic::run_tests!();
}
