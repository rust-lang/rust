//@ aux-build:rustdoc-trait-object-impl.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate rustdoc_trait_object_impl;

//@ has issue_32881/trait.Bar.html
//@ has - '//h3[@class="code-header"]' "impl<'a> dyn Bar"
//@ has - '//h3[@class="code-header"]' "impl<'a> Debug for dyn Bar"

pub use rustdoc_trait_object_impl::Bar;
