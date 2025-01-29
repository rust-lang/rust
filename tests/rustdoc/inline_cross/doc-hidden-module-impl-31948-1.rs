// https://github.com/rust-lang/rust/issues/31948
#![crate_name="foobar"]

//@ aux-build:rustdoc-nonreachable-impls.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

//@ has foobar/struct.Wobble.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bark for'
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'Woof for'
//@ !has - '//*[@class="impl"]//h3[@class="code-header"]' 'Bar for'
//@ !has - '//*[@class="impl"]//h3[@class="code-header"]' 'Qux for'
pub use rustdoc_nonreachable_impls::hidden::Wobble;

//@ has foobar/trait.Bark.html
//@ has - '//h3[@class="code-header"]' 'for Foo'
//@ has - '//h3[@class="code-header"]' 'for Wobble'
//@ !has - '//h3[@class="code-header"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Bark;

//@ has foobar/trait.Woof.html
//@ has - '//h3[@class="code-header"]' 'for Foo'
//@ has - '//h3[@class="code-header"]' 'for Wobble'
//@ !has - '//h3[@class="code-header"]' 'for Wibble'
pub use rustdoc_nonreachable_impls::Woof;

//@ !has foobar/trait.Bar.html
//@ !has foobar/trait.Qux.html
