//! Tests cleanup of a temporary `Box` rvalue passed as a mutable reference.
//!
//! - Issue: <https://github.com/rust-lang/rust/issues/7972>.

//@ run-pass

fn foo(x: &mut Box<u8>) {
    *x = Box::new(5);
}

pub fn main() {
    foo(&mut Box::new(4));
}
