// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948/struct.Foo.html
// @has - '//*[@class="impl has-srclink"]//h3' 'Bark for'
// @has - '//*[@class="impl has-srclink"]//h3' 'Woof for'
// @!has - '//*[@class="impl has-srclink"]//h3' 'Bar for'
// @!has - '//*[@class="impl"]//h3' 'Qux for'
pub use rustdoc_nonreachable_impls::Foo;

// @has issue_31948/trait.Bark.html
// @has - '//h3' 'for Foo'
// @!has - '//h3' 'for Wibble'
// @!has - '//h3' 'for Wobble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948/trait.Woof.html
// @has - '//h3' 'for Foo'
// @!has - '//h3' 'for Wibble'
// @!has - '//h3' 'for Wobble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948/trait.Bar.html
// @!has issue_31948/trait.Qux.html
// @!has issue_31948/struct.Wibble.html
// @!has issue_31948/struct.Wobble.html
