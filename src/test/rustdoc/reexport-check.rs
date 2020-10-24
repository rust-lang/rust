// aux-build:reexport-check.rs
#![crate_name = "foo"]

extern crate reexport_check;

// @!has 'foo/index.html' '//code' 'pub use self::i32;'
// @has 'foo/index.html' '//tr[@class="module-item"]' 'i32'
// @has 'foo/i32/index.html'
#[allow(deprecated, deprecated_in_future)]
pub use std::i32;
// @!has 'foo/index.html' '//code' 'pub use self::string::String;'
// @has 'foo/index.html' '//tr[@class="module-item"]' 'String'
pub use std::string::String;

// @has 'foo/index.html' '//td[@class="docblock-short"]' 'Docs in original'
// this is a no-op, but shows what happens if there's an attribute that isn't a doc-comment
#[doc(inline)]
pub use reexport_check::S;
