#![warn(clippy::large_include_file)]

// Good
const GOOD_INCLUDE_BYTES: &[u8; 68] = include_bytes!("../../ui/author.rs");
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

#[doc = include_str!("too_big.txt")] //~ large_include_file
fn main() {}
