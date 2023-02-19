// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948_1/struct.Wobble.html
// @has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bark for'
// @has - '//*[@class="impl"]//h3[@class="code-header"]' 'Woof for'
// @!has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bar for'
// @!has - '//*[@class="impl"]//h3[@class="code-header"]' 'Qux for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

// @has issue_31948_1/trait.Bark.html
// @has - '//h3[@class="code-header"]' 'for Foo'
// @has - '//h3[@class="code-header"]' 'for Wobble'
// @!has - '//h3[@class="code-header"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948_1/trait.Woof.html
// @has - '//h3[@class="code-header"]' 'for Foo'
// @has - '//h3[@class="code-header"]' 'for Wobble'
// @!has - '//h3[@class="code-header"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948_1/trait.Bar.html
// @!has issue_31948_1/trait.Qux.html
