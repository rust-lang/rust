//@ run-pass

#![allow(warnings)]

//! Tests type inference fallback to `!` (never type) in `Option` context.
//!
//! Regression test for issues:
//! - https://github.com/rust-lang/rust/issues/39808
//! - https://github.com/rust-lang/rust/issues/39984
//!
//! Checks that when creating `Option` from a diverging expression,
//! the type parameter defaults to `!` without requiring explicit annotations.

fn main() {
    // This expression diverges, so its type is `!`
    let c = Some({ return; });
    // The type of `c` is inferred as `Option<!>`
    c.unwrap();
}
