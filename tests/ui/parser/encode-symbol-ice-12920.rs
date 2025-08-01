//! Regression test for https://github.com/rust-lang/rust/issues/12920

//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

pub fn main() {
    panic!();
    println!("{}", 1);
}
