#![crate_type = "rlib"]

// Suggestions for range patterns should not perform span manipulations that
// assume the range token is ASCII, because it could have been recovered from
// similar-looking Unicode characters.
//
// Regression test for <https://github.com/rust-lang/rust/issues/155799>.

// FIXME: The ICE is fixed in a subsequent commit.
//@ known-bug: #155799
//@ failure-status: 101

// These dots are U+00B7 MIDDLE DOT, not an ASCII period.
fn dot_dot_dot() { ··· }
