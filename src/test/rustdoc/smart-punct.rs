#![crate_name = "foo"]

//! This is the "start" of the 'document'! How'd you know that "it's" the start?
//!
//! # Header with "smart punct'"
//!
//! [link with "smart punct'" -- yessiree!][]
//!
//! [link with "smart punct'" -- yessiree!]: https://www.rust-lang.org

// @has "foo/index.html" "//p" "This is the “start” of the ‘document’! How’d you know that “it’s” the start?"
// @has "foo/index.html" "//h1" "Header with “smart punct’”"
// @has "foo/index.html" '//a[@href="https://www.rust-lang.org"]' "link with “smart punct’” – yessiree!"
