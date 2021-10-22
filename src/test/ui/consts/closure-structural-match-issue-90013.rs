// Regression test for issue 90013.
// check-pass
#![allow(incomplete_features)]
#![feature(inline_const)]

fn main() {
    const { || {} };
}
