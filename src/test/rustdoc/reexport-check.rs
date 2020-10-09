#![crate_name = "foo"]

// @!has 'foo/index.html' '//code' 'pub use self::i32;'
// @has 'foo/index.html' '//tr[@class="module-item"]' 'i32'
// @has 'foo/i32/index.html'
pub use std::i32;
// @!has 'foo/index.html' '//code' 'pub use self::string::String;'
// @has 'foo/index.html' '//tr[@class="module-item"]' 'String'
pub use std::string::String;
