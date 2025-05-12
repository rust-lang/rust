//@error-in-other-file:
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::no_effect)]

#[path = "auxiliary/ice-7868-aux.rs"]
mod zero;

fn main() {}
