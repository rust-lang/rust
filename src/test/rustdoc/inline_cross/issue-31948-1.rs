// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948_1/struct.Wobble.html
// @has - '//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' 'Bark for'
// @has - '//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' 'Woof for'
// @!has - '//*[@class="impl"]//h3[@class="code-header in-band"]' 'Bar for'
// @!has - '//*[@class="impl"]//h3[@class="code-header in-band"]' 'Qux for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

// @has issue_31948_1/trait.Bark.html
// @has - '//h3[@class="code-header in-band"]' 'for Foo'
// @has - '//h3[@class="code-header in-band"]' 'for Wobble'
// @!has - '//h3[@class="code-header in-band"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948_1/trait.Woof.html
// @has - '//h3[@class="code-header in-band"]' 'for Foo'
// @has - '//h3[@class="code-header in-band"]' 'for Wobble'
// @!has - '//h3[@class="code-header in-band"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948_1/trait.Bar.html
// @!has issue_31948_1/trait.Qux.html
