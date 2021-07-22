// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948_1/struct.Wobble.html
// @has - '//*[@class="impl has-srclink"]//h3' 'Bark for'
// @has - '//*[@class="impl has-srclink"]//h3' 'Woof for'
// @!has - '//*[@class="impl"]//h3' 'Bar for'
// @!has - '//*[@class="impl"]//h3' 'Qux for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

// @has issue_31948_1/trait.Bark.html
// @has - '//h3' 'for Foo'
// @has - '//h3' 'for Wobble'
// @!has - '//h3' 'for Wibble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948_1/trait.Woof.html
// @has - '//h3' 'for Foo'
// @has - '//h3' 'for Wobble'
// @!has - '//h3' 'for Wibble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948_1/trait.Bar.html
// @!has issue_31948_1/trait.Qux.html
