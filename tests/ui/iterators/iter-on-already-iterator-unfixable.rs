//! Tests for the targeted diagnostic that suggests removing redundant `.iter()` calls.
//! This file focuses on "unfixable" cases where a comment is present *inside* the
//! portion being deleted (e.g. `. /* comment */ iter()`).
//!
//! These are marked as `MaybeIncorrect` to ensure `cargo fix` does not
//! automatically delete user comments.
#![allow(unused)]

#[rustfmt::skip]
fn main() {
    let receiver = vec![1, 2, 3].into_iter();

    // A comment containing a dot shouldn't trick the span parser.
    // The comment is BETWEEN the dot and the method name, so it would be deleted.
    let _ = receiver . /* . */ iter();
    //~^ ERROR no method named `iter`
    //~| HELP consider removing the `.iter()` call
}
