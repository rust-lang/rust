//@ aux-build: incoherent_impl1.rs
//@ aux-build: incoherent_impl2.rs
//@ build-aux-docs
//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

extern crate incoherent_impl1;
extern crate incoherent_impl2;

pub use incoherent_impl2::Error;

//@ has 'src/foo/incoherent_impl-inlined.rs.html'
//@ has - '//pre//a[@href="../../foo/struct.Error.html#method.new"]' 'new'

//@ has 'foo/struct.Error.html'
//@ has - '//*[@id="method.new"]' 'pub fn new() -> Error'

fn foo() {
    let x = Error::new();
}
