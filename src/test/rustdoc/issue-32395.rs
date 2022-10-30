// aux-build:variant-struct.rs
// build-aux-docs
// ignore-cross-compile

// @has variant_struct/enum.Foo.html
// @!hasraw - 'pub qux'
// @!hasraw - 'pub(crate) qux'
// @!hasraw - 'pub Bar'
extern crate variant_struct;

// @has issue_32395/enum.Foo.html
// @!hasraw - 'pub qux'
// @!hasraw - 'pub(crate) qux'
// @!hasraw - 'pub Bar'
pub use variant_struct::Foo;
