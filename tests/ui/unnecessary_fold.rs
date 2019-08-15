// run-rustfix

#![allow(dead_code)]

/// Calls which should trigger the `UNNECESSARY_FOLD` lint
fn unnecessary_fold() {
    // Can be replaced by .any
    let _ = (0..3).fold(false, |acc, x| acc || x > 2);
    // Can be replaced by .all
    let _ = (0..3).fold(true, |acc, x| acc && x > 2);
    // Can be replaced by .sum
    let _: i32 = (0..3).fold(0, |acc, x| acc + x);
    // Can be replaced by .product
    let _: i32 = (0..3).fold(1, |acc, x| acc * x);
}

/// Should trigger the `UNNECESSARY_FOLD` lint, with an error span including exactly `.fold(...)`
fn unnecessary_fold_span_for_multi_element_chain() {
    let _: bool = (0..3).map(|x| 2 * x).fold(false, |acc, x| acc || x > 2);
}

/// Calls which should not trigger the `UNNECESSARY_FOLD` lint
fn unnecessary_fold_should_ignore() {
    let _ = (0..3).fold(true, |acc, x| acc || x > 2);
    let _ = (0..3).fold(false, |acc, x| acc && x > 2);
    let _ = (0..3).fold(1, |acc, x| acc + x);
    let _ = (0..3).fold(0, |acc, x| acc * x);
    let _ = (0..3).fold(0, |acc, x| 1 + acc + x);

    // We only match against an accumulator on the left
    // hand side. We could lint for .sum and .product when
    // it's on the right, but don't for now (and this wouldn't
    // be valid if we extended the lint to cover arbitrary numeric
    // types).
    let _ = (0..3).fold(false, |acc, x| x > 2 || acc);
    let _ = (0..3).fold(true, |acc, x| x > 2 && acc);
    let _ = (0..3).fold(0, |acc, x| x + acc);
    let _ = (0..3).fold(1, |acc, x| x * acc);

    let _ = [(0..2), (0..3)].iter().fold(0, |a, b| a + b.len());
    let _ = [(0..2), (0..3)].iter().fold(1, |a, b| a * b.len());
}

/// Should lint only the line containing the fold
fn unnecessary_fold_over_multiple_lines() {
    let _ = (0..3)
        .map(|x| x + 1)
        .filter(|x| x % 2 == 0)
        .fold(false, |acc, x| acc || x > 2);
}

fn main() {}
