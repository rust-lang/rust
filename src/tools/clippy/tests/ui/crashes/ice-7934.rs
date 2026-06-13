//@ check-pass

#![warn(clippy::undocumented_unsafe_blocks)]

#[path = "auxiliary/ice-7934-aux.rs"]
mod zero;

fn main() {}
