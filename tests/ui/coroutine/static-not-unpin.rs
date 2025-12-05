//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver --diagnostic-width=300
//@[current] compile-flags: --diagnostic-width=300

#![feature(coroutines, stmt_expr_attributes)]

//@ normalize-stderr: "std::pin::Unpin" -> "std::marker::Unpin"

use std::marker::Unpin;

fn assert_unpin<T: Unpin>(_: T) {}

fn main() {
    let mut coroutine = #[coroutine]
    static || {
        yield;
    };
    assert_unpin(coroutine); //~ ERROR E0277
}
