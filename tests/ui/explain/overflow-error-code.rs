// Check that we don't crash on error codes exceeding our internal limit.
// issue: <https://github.com/rust-lang/rust/issues/140647>
//@ compile-flags: --explain E10000
//~? ERROR: E10000 is not a valid error code
