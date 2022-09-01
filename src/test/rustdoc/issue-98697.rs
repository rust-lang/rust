// aux-build:issue-98697-reexport-with-anonymous-lifetime.rs
// ignore-cross-compile

// When reexporting a function with a HRTB with anonymous lifetimes,
// make sure the anonymous lifetimes are not rendered.
//
// https://github.com/rust-lang/rust/issues/98697

extern crate issue_98697_reexport_with_anonymous_lifetime;

// @has issue_98697/fn.repro.html '//pre[@class="rust fn"]/code' 'fn repro<F>()where F: Fn(&str)'
// @!has issue_98697/fn.repro.html '//pre[@class="rust fn"]/code' 'for<'
pub use issue_98697_reexport_with_anonymous_lifetime::repro;

// @has issue_98697/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header in-band"]' 'impl MyTrait<&Extra> for Extra'
// @!has issue_98697/struct.Extra.html '//div[@id="trait-implementations-list"]//h3[@class="code-header in-band"]' 'impl<'
pub use issue_98697_reexport_with_anonymous_lifetime::Extra;
