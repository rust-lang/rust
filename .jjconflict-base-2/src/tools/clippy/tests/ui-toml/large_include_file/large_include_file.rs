#![warn(clippy::large_include_file)]
#![allow(clippy::literal_string_with_formatting_args)]

// Good
const GOOD_INCLUDE_BYTES: &[u8; 84] = include_bytes!("../../ui/author.rs");
const GOOD_INCLUDE_STR: &str = include_str!("../../ui/author.rs");

#[allow(clippy::large_include_file)]
const ALLOWED_TOO_BIG_INCLUDE_BYTES: &[u8; 654] = include_bytes!("too_big.txt");
#[allow(clippy::large_include_file)]
const ALLOWED_TOO_BIG_INCLUDE_STR: &str = include_str!("too_big.txt");

// Bad
const TOO_BIG_INCLUDE_BYTES: &[u8; 654] = include_bytes!("too_big.txt");
//~^ large_include_file
const TOO_BIG_INCLUDE_STR: &str = include_str!("too_big.txt");
//~^ large_include_file

#[doc = include_str!("too_big.txt")]
//~^ large_include_file
// Should not lint!
// Regression test for <https://github.com/rust-lang/rust-clippy/issues/13670>.
#[doc = include_str!("empty.txt")]
fn main() {}
