//! Test that a diverging function as the final expression in a block does not
//! raise an 'unreachable code' lint.

//@ check-pass
#![deny(unreachable_code)]

use std::process::ExitCode;

enum Never {}

fn make_never() -> Never {
    loop {}
}

fn func() {
    make_never();
}

fn block() {
    {
        make_never();
    }
}

fn branchy() {
    if false {
        make_never();
    } else {
        make_never();
    }
}

// Regression test for https://github.com/rust-lang/rust/issues/152559.
// The final expression is unreachable at runtime, but it cannot be removed
// because it supplies the function's required return type.
fn required_return_value() -> ExitCode {
    make_never();
    ExitCode::FAILURE
}

fn main() {
    func();
    block();
}
