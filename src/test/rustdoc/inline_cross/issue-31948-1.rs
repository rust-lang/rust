// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948_1/struct.Wobble.html
// @has - '//*[@class="impl"]//code' 'Bark for'
// @has - '//*[@class="impl"]//code' 'Woof for'
// @!has - '//*[@class="impl"]//code' 'Bar for'
// @!has - '//*[@class="impl"]//code' 'Qux for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

// @has issue_31948_1/trait.Bark.html
// @has - '//code' 'for Foo'
// @has - '//code' 'for Wobble'
// @!has - '//code' 'for Wibble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948_1/trait.Woof.html
// @has - '//code' 'for Foo'
// @has - '//code' 'for Wobble'
// @!has - '//code' 'for Wibble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948_1/trait.Bar.html
// @!has issue_31948_1/trait.Qux.html
