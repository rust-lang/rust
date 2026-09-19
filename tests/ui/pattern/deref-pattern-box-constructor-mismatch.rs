//@ run-pass
//! Test that `Box { .. }` is treated like a wildcard.

#![feature(deref_patterns)]

fn main() {
    match Box::new(0) {
        deref!(_) => {}
        Box { .. } => {} //~ WARN unreachable pattern
    }
}
