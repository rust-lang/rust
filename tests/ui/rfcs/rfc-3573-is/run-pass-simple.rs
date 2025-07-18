//! Simple tests for matching and binding with `is` expressions.
//@ run-pass
//@ edition: 2024
//@ aux-crate: is_macro=is-macro.rs
#![feature(builtin_syntax)]
use is_macro::is;

// Test `is` used as `let` in `if` conditions.
fn test_if() {
    for test_in in [None, Some(3), Some(4)] {
        let test_if_expr = if is!(test_in is Some(x)) && x > 3 {
            // `x` is bound in the success block
            x
        } else {
            0
        };
        let test_guard = match test_in {
            x if is!(x is Some(y)) && y > 3 => {
                // `y` is bound in the arm body
                y
            }
            _ => 0,
        };
        let test_expected = if let Some(x) = test_in && x > 3 {
            x
        } else {
            0
        };
        assert_eq!(test_if_expr, test_expected);
        assert_eq!(test_guard, test_expected);
    }
}

// Test `is` used as `let` in `while` conditions.
fn test_while() {
    let mut count = 0;
    let mut opt = Some(3u8);
    while is!(opt is Some(x)) {
        // `x` is bound in the loop body
        count += 1;
        opt = x.checked_sub(1);
    }
    assert_eq!(count, 4);
}

// Test `is` not directly in an `if` or `while` condition.
fn test_elsewhere() {
    for test_in in [None, Some(None), Some(Some(3)), Some(Some(4))] {
        let test_is = is!(test_in is Some(x)) && is!(x is Some(y)) && y > 3;
        let test_expected =
            if let Some(x) = test_in && let Some(y) = x && y > 3 { true } else { false };
        assert_eq!(test_is, test_expected);
    }
}

fn main() {
    test_if();
    test_while();
    test_elsewhere();
}
