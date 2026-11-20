//! Regression test for https://github.com/rust-lang/rust/issues/12033

//@ run-pass
use std::cell::RefCell;

fn main() {
    let x = RefCell::new(0);
    if *x.borrow() == 0 {} else {}
}
