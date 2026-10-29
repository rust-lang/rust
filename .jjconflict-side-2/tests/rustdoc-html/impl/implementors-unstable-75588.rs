//@ aux-build:realcore.rs
//@ aux-build:real_gimli.rs

// Ensure unstably exported traits have their Implementors sections.
// https://github.com/rust-lang/rust/issues/75588

#![crate_name = "foo"]
#![feature(extremely_unstable_foo)]

extern crate realcore;
extern crate real_gimli;

// issue #74672
//@ !has foo/trait.Deref.html '//*[@id="impl-Deref-for-EndianSlice"]//h3[@class="code-header"]' 'impl Deref for EndianSlice'
pub use realcore::Deref;

//@ has foo/trait.Join.html '//*[@id="impl-Join-for-Foo"]//h3[@class="code-header"]' 'impl Join for Foo'
pub use realcore::Join;
