#![crate_name = "foo"]

//! This is the "start" of the 'document'! How'd you know that "it's" the start?
//!
//! # Header with "smart punct'"
//!
//! [link with "smart punct'" -- yessiree!][]
//!
//! [link with "smart punct'" -- yessiree!]: https://www.rust-lang.org
//!
//! # Code should not be smart-punct'd
//!
//! `this inline code -- it shouldn't have "smart punct"`
//!
//! ```
//! let x = "don't smart-punct me -- please!";
//! ```
//!
//! ```text
//! I say "don't smart-punct me -- please!"
//! ```

// @has "foo/index.html" "//p" "This is the “start” of the ‘document’! How’d you know that “it’s” the start?"
// @has "foo/index.html" "//h2" "Header with “smart punct’”"
// @has "foo/index.html" '//a[@href="https://www.rust-lang.org"]' "link with “smart punct’” – yessiree!"
// @has "foo/index.html" '//code' "this inline code -- it shouldn't have \"smart punct\""
// @has "foo/index.html" '//pre' "let x = \"don't smart-punct me -- please!\";"
// @has "foo/index.html" '//pre' "I say \"don't smart-punct me -- please!\""
