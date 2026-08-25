//@ check-pass
// This is a regression test for <https://github.com/rust-lang/rust-clippy/issues/13183>.
// It should not warn on missing backticks on footnote references.

#![warn(clippy::doc_markdown)]
// Should not warn!
//! Here's a footnote[^example_footnote_identifier]
//!
//! [^example_footnote_identifier]: This is merely an example.

fn main() {}
