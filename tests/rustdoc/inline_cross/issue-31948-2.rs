// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948_2/struct.Wobble.html
// @has - '//*[@class="impl"]//h3[@class="code-header"]' 'Qux for'
// @has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bark for'
// @has - '//*[@class="impl"]//h3[@class="code-header"]' 'Woof for'
// @!has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bar for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

// @has issue_31948_2/trait.Qux.html
// @has - '//h3[@class="code-header"]' 'for Foo'
// @has - '//h3[@class="code-header"]' 'for Wobble'
pub use rustdoc_nonreachable_impls::hidden::Qux;

// @!has issue_31948_2/trait.Bar.html
// @!has issue_31948_2/trait.Woof.html
// @!has issue_31948_2/trait.Bark.html
