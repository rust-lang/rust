// Test to ensure that we can link to the current crate by using its name.
//@ aux-build:intra-link-current-crate.rs
//@ build-aux-docs

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "bar"]

//@ has 'bar/index.html'
//@ has - '//*[@class="docblock"]//a[@href="index.html"]' 'bar'
//! [`bar`]

extern crate tarte;

//@ has 'bar/struct.Tarte.html'
//@ has - '//*[@class="docblock"]//a[@href="../tarte/index.html"]' 'tarte'
#[doc(inline)]
pub use tarte::Tarte;

pub mod foo {
    //@ has 'bar/foo/fn.tadam.html'
    //@ has - '//*[@class="docblock"]//a[@href="../index.html"]' 'bar'
    /// [`bar`]
    pub fn tadam() {}
}
