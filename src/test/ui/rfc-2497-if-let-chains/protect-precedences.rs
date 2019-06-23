// run-pass

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn main() {
    let x: bool;
    // This should associate as: `(x = (true && false));`.
    x = true && false;
    assert!(!x);

    fn _f1() -> bool {
        // Should associate as `(let _ = (return (true && false)))`.
        if let _ = return true && false {};
    }
    assert!(!_f1());
}
