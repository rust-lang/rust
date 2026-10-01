//! Regression test for <https://github.com/rust-lang/rust/issues/48944>.
//!
//! The first token parsed from a string with `FromStr` for `TokenStream` used to report a
//! different source file than the tokens after it.

//@ check-pass
//@ proc-macro: span-first-token-file-48944.rs

extern crate span_first_token_file_48944;

span_first_token_file_48944::check_first_token_file!();

fn main() {}
