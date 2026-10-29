//! Regression test for https://github.com/rust-lang/rust/issues/19398
//@ check-pass

trait T {
    unsafe extern "Rust" fn foo(&self);
}

impl T for () {
    unsafe extern "Rust" fn foo(&self) {}
}

fn main() {}
