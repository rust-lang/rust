//@ compile-flags: --edition=2024 -Z unstable-options
//@ aux-build: metavar_2021.rs
//@ run-pass

// This test captures the behavior of macro-generating-macros with fragment
// specifiers across edition boundaries.

#![feature(expr_fragment_specifier_2024)]
#![feature(macro_metavar_expr)]
#![allow(incomplete_features)]

extern crate metavar_2021;

use metavar_2021::{is_expr_from_2021, make_matcher};

make_matcher!(is_expr_from_2024, expr, $);

fn main() {
    let from_2021 = is_expr_from_2021!(const { 0 });
    dbg!(from_2021);
    let from_2024 = is_expr_from_2024!(const { 0 });
    dbg!(from_2024);

    // These capture the current, empirically determined behavior.
    // It's not clear whether this is the desired behavior.
    assert!(!from_2021);
    assert!(!from_2024);
}
