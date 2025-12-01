//@ check-pass
//! Regression test for issue #146445.
//!
//! This test ensures that liveness linting works correctly with guard patterns
//! and doesn't cause an ICE ("no successor") when using variables without
//! underscore prefixes in guard expressions.
//!
//! Fixed by #142390 which moved liveness analysis to MIR.

#![feature(guard_patterns)]
#![expect(incomplete_features)]

fn main() {
    // This used to ICE before liveness analysis was moved to MIR.
    // The variable `main` (without underscore prefix) would trigger
    // liveness linting which wasn't properly implemented for guard patterns.
    match (0,) {
        (_ if { let main = false; main },) => {}
        _ => {}
    }
}
