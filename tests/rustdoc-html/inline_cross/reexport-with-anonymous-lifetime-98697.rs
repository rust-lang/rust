//@ aux-build:reexport-with-anonymous-lifetime-98697.rs
//@ ignore-cross-compile
#![crate_name = "foo"]

// When reexporting a function with a HRTB with anonymous lifetimes,
// make sure the anonymous lifetimes are not rendered.
//
// https://github.com/rust-lang/rust/issues/98697

extern crate reexport_with_anonymous_lifetime_98697;

//@ has foo/fn.repro.html '//pre[@class="rust item-decl"]/code' 'fn repro<F>()where F: Fn(&str)'
//@ !has foo/fn.repro.html '//pre[@class="rust item-decl"]/code' 'for<'
pub use reexport_with_anonymous_lifetime_98697::repro;

//@ has foo/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header"]' 'impl MyTrait<&Extra> for Extra'
//@ !has foo/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header"]' 'impl<'
pub use reexport_with_anonymous_lifetime_98697::Extra;
