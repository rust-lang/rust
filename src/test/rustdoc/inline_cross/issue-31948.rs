// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948/struct.Foo.html
// @has - '//*[@class="impl"]//code' 'Bark for'
// @has - '//*[@class="impl"]//code' 'Woof for'
// @!has - '//*[@class="impl"]//code' 'Bar for'
// @!has - '//*[@class="impl"]//code' 'Qux for'
pub use rustdoc_nonreachable_impls::Foo;

// @has issue_31948/trait.Bark.html
// @has - '//code' 'for Foo'
// @!has - '//code' 'for Wibble'
// @!has - '//code' 'for Wobble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948/trait.Woof.html
// @has - '//code' 'for Foo'
// @!has - '//code' 'for Wibble'
// @!has - '//code' 'for Wobble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948/trait.Bar.html
// @!has issue_31948/trait.Qux.html
// @!has issue_31948/struct.Wibble.html
// @!has issue_31948/struct.Wobble.html
