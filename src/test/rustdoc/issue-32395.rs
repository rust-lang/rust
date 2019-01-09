// aux-build:variant-struct.rs
// build-aux-docs
// ignore-cross-compile

// @has variant_struct/enum.Foo.html
// @!has - 'pub qux'
// @!has - 'pub Bar'
extern crate variant_struct;

// @has issue_32395/enum.Foo.html
// @!has - 'pub qux'
// @!has - 'pub Bar'
pub use variant_struct::Foo;
