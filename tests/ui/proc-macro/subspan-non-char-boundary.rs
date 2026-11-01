// Regression test for https://github.com/rust-lang/rust/issues/156049
//
// Tests that `Literal::subspan` returns `None` when the byte range does not
// fall on a UTF-8 char boundary in the source.

//@ check-pass
//@ proc-macro: subspan-non-char-boundary.rs

extern crate subspan_non_char_boundary;

use subspan_non_char_boundary::check_non_char_boundary;

fn main() {
    check_non_char_boundary!("🦀");
}
