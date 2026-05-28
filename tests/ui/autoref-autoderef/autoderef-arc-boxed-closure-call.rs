//! regression test for <https://github.com/rust-lang/rust/issues/21306>
//@ run-pass

use std::sync::Arc;

fn main() {
    let x = 5;
    let command = Arc::new(Box::new(|| x * 2));
    assert_eq!(command(), 10);
}
