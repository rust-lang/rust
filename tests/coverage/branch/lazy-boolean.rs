#![feature(coverage_attribute)]
//@ edition: 2021
//@ compile-flags: -Zcoverage-options=branch
//@ llvm-cov-flags: --show-branches=count

// Tests for branch coverage of the lazy boolean operators `&&` and `||`,
// as ordinary expressions that aren't part of an `if` condition or similar.

use core::hint::black_box;

// Helper macro to prevent start-of-function spans from being merged into
// spans on the lines we care about.
macro_rules! no_merge {
    () => {
        for _ in 0..1 {}
    };
}

fn branch_and(a: bool, b: bool) {
    no_merge!();

    //      |13  |18 (no branch)
    let c = a && b;
    black_box(c);
}

fn branch_or(a: bool, b: bool) {
    no_merge!();

    //      |13  |18 (no branch)
    let c = a || b;
    black_box(c);
}

// Test for chaining one operator several times.
fn chain(x: u32) {
    no_merge!();

    //      |13      |22      |31      |40 (no branch)
    let c = x > 1 && x > 2 && x > 4 && x > 8;
    black_box(c);

    //      |13      |22      |31      |40 (no branch)
    let d = x < 1 || x < 2 || x < 4 || x < 8;
    black_box(d);
}

// Test for nested combinations of different operators.
fn nested_mixed(x: u32) {
    no_merge!();

    //       |14      |23         |35      |44 (no branch)
    let c = (x < 4 || x >= 9) && (x < 2 || x >= 10);
    black_box(c);

    //       |14      |23        |34       |44 (no branch)
    let d = (x < 4 && x < 1) || (x >= 8 && x >= 10);
    black_box(d);
}

#[coverage(off)]
fn main() {
    // Use each set of arguments (2^n) times, so that each combination has a
    // unique sum, and we can use those sums to verify expected control flow.
    // 1x (false, false)
    // 2x (false, true)
    // 4x (true, false)
    // 8x (true, true)
    for a in [false, true, true, true, true] {
        for b in [false, true, true] {
            branch_and(a, b);
            branch_or(a, b);
        }
    }

    for x in 0..16 {
        chain(x);
        nested_mixed(x);
    }
}
