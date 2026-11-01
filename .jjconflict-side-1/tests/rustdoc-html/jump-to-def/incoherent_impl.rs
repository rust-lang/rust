// This test ensures that we generate correct links when we target a method implemented
// with `#[rustc_allow_incoherent_impl]` in a different crate than where the type is defined.
// Regression test for <https://github.com/rust-lang/rust/issues/163112>.

//@ aux-build: incoherent_impl1.rs
//@ build-aux-docs
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

extern crate incoherent_impl1 as bar;

//@ has 'src/foo/incoherent_impl.rs.html'
//@ has - '//pre//a[@href="../../incoherent_impl1/error/struct.Error.html#method.new"]' 'new'
//@ has - '//pre//a[@href="../../incoherent_impl2/struct.Error.html"]' 'Error'

// Now we check that the target files exist and contain the information we want.
//@ has 'incoherent_impl1/error/struct.Error.html'
//@ has - '//*[@id="method.new"]' 'pub fn new() -> Self'
//@ has 'incoherent_impl2/struct.Error.html'

fn foo() {
    let x = bar::error::Error::new();
}
