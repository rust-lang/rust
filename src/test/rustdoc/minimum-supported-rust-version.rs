// compile-flags:-Zunstable-options --minimum-supported-rust-version 1.48.0

#![crate_name = "foo"]

// @has 'foo/index.html'
// @has - '//*[@id="main-content"]/*[@class="extra-info"]' 'ⓘ The minimum supported Rust version for this crate is: 1.48.0.'
// @has - '//*[@class="rustdoc-toggle top-doc"]/*[@class="docblock"]' 'This crate is awesome.'

//! This crate is awesome.

// We check that the minimum supported rust version is only on the crate page.
// @has 'foo/struct.Foo.html'
// @!has - '//*[@class="extra-info"]'
pub struct Foo;

// @has 'foo/bar/index.html'
// @!has - '//*[@class="extra-info"]'
pub mod bar {}
