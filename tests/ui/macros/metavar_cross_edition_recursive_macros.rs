//@ compile-flags: --edition=2024 -Z unstable-options
//@ aux-build: metavar_2018.rs
//@ run-pass

// This test captures the behavior of macro-generating-macros with fragment
// specifiers across edition boundaries.

#![feature(expr_fragment_specifier_2024)]
#![feature(macro_metavar_expr)]
#![allow(incomplete_features)]

extern crate metavar_2018;

use metavar_2018::{is_expr_from_2018, is_pat_from_2018, make_matcher};

make_matcher!(is_expr_from_2024, expr, $);
make_matcher!(is_pat_from_2024, pat, $);

fn main() {
    // Check expr
    let from_2018 = is_expr_from_2018!(const { 0 });
    dbg!(from_2018);
    let from_2024 = is_expr_from_2024!(const { 0 });
    dbg!(from_2024);

    // These capture the current, empirically determined behavior.
    // It's not clear whether this is the desired behavior.
    assert!(!from_2018);
    assert!(!from_2024);

    // Check pat
    let from_2018 = is_pat_from_2018!(A | B);
    dbg!(from_2018);
    let from_2024 = is_pat_from_2024!(A | B);
    dbg!(from_2024);

    // These capture the current, empirically determined behavior.
    // It's not clear whether this is the desired behavior.
    assert!(!from_2018);
    assert!(!from_2024);
}
