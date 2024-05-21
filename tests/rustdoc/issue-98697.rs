//@ aux-build:issue-98697-reexport-with-anonymous-lifetime.rs
//@ ignore-cross-compile
#![crate_name = "foo"]

// When reexporting a function with a HRTB with anonymous lifetimes,
// make sure the anonymous lifetimes are not rendered.
//
// https://github.com/rust-lang/rust/issues/98697

extern crate issue_98697_reexport_with_anonymous_lifetime;

// @has foo/fn.repro.html '//pre[@class="rust item-decl"]/code' 'fn repro<F>()where F: Fn(&str)'
// @!has foo/fn.repro.html '//pre[@class="rust item-decl"]/code' 'for<'
pub use issue_98697_reexport_with_anonymous_lifetime::repro;

// @has foo/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header"]' 'impl MyTrait<&Extra> for Extra'
// @!has foo/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header"]' 'impl<'
pub use issue_98697_reexport_with_anonymous_lifetime::Extra;
