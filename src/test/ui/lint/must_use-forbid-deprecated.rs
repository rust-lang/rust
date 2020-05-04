//! Tests that the lowering of `#[must_use]` compiles in presence of `#![forbid(deprecated)]`.

// check-pass

#![deny(unused_must_use)]
#![forbid(deprecated)]

#[must_use]
struct S;

fn main() {}
