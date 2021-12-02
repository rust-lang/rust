// Regression test for issue 90013.
// check-pass
#![feature(inline_const)]

fn main() {
    const { || {} };
}
