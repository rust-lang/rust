// Regression test for <https://github.com/rust-lang/rust/issues/53724>.

//@ aux-build:glob-of-reexports-53724.rs
//@ build-aux-docs
//@ ignore-cross-compile

#![crate_name = "foo"]

extern crate inner;

//@ has foo/trait.Serialize.html
//@ has foo/trait.Serializer.html
//@ has foo/trait.Deserialize.html
//@ has foo/index.html '//a[@class="trait"]' 'Serialize'
//@ has foo/index.html '//a[@class="trait"]' 'Serializer'
//@ has foo/index.html '//a[@class="trait"]' 'Deserialize'
pub use inner::*;
