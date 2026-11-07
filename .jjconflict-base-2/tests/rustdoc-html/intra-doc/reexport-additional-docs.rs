//@ aux-build:intra-link-reexport-additional-docs.rs
//@ build-aux-docs
#![crate_name = "foo"]
extern crate inner;

//@ has foo/struct.Inner.html '//a[@href="fn.with_code.html"]' 'crate::with_code'
/// [crate::with_code]
//@ has - '//a[@href="fn.with_code.html"]' 'different text'
/// [different text][with_code]
//@ has - '//a[@href="fn.me_too.html"]' 'me_too'
#[doc = "[me_too]"]
//@ has - '//a[@href="fn.me_three.html"]' 'reference link'
/// This [reference link]
#[doc = "has an attr in the way"]
///
/// [reference link]: me_three
// Should still resolve links from the original module in that scope
//@ has - '//a[@href="../inner/fn.f.html"]' 'f()'
pub use inner::Inner;

pub fn with_code() {}
pub fn me_too() {}
pub fn me_three() {}
