//! Regression test for <https://github.com/rust-lang/rust/issues/16922>.
//@ check-pass

use std::any::Any;

fn foo(_: &u8) {
}

fn main() {
    let _ = &foo as &dyn Any;
}
