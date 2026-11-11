#![crate_name = "foo"]

//! # Heading with [a link](https://a.com) inside
//!
//! And even with
//!
//! ## [multiple](https://b.com) [links](https://c.com)
//!
//! !

//@ has 'foo/index.html'
//@ has - '//h2/a[@href="https://a.com"]' 'a link'
//@ has - '//h3/a[@href="https://b.com"]' 'multiple'
//@ has - '//h3/a[@href="https://c.com"]' 'links'
