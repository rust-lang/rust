//! Smoke test: dereferencing boxed zero-sized types (ZSTs) should not crash.
//!
//! Originally a regression test of github.com/rust-lang/rust/issues/13360
//! but repurposed for a smoke test.

//@ run-pass

pub fn main() {
    let _: () = *Box::new(());
}
