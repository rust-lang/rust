// aux-build:reexport-check.rs
#![crate_name = "foo"]

extern crate reexport_check;

// @!has 'foo/index.html' '//code' 'pub use self::i32;'
// @has 'foo/i32/index.html'
#[allow(deprecated, deprecated_in_future)]
pub use std::i32;
// @!has 'foo/index.html' '//code' 'pub use self::string::String;'
// @has 'foo/index.html' '//div[@class="item-left"]' 'String'
pub use std::string::String;

// i32 is deprecated, String is not
// @count 'foo/index.html' '//span[@class="stab deprecated"]' 1

// @has 'foo/index.html' '//div[@class="item-right docblock-short"]' 'Docs in original'
// this is a no-op, but shows what happens if there's an attribute that isn't a doc-comment
#[doc(inline)]
pub use reexport_check::S;
