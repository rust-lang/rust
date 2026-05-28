// https://github.com/rust-lang/rust/issues/32881
#![crate_name="foobar"]

//@ aux-build:rustdoc-trait-object-impl.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate rustdoc_trait_object_impl;

//@ has foobar/trait.Bar.html
//@ has - '//h3[@class="code-header"]' "impl<'a> dyn Bar"
//@ has - '//h3[@class="code-header"]' "impl<'a> Debug for dyn Bar"

pub use rustdoc_trait_object_impl::Bar;
