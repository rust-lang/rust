// aux-build:realcore.rs
// aux-build:real_gimli.rs

// Ensure unstably exported traits have their Implementors sections.

#![crate_name = "foo"]
#![feature(extremely_unstable_foo)]

extern crate realcore;
extern crate real_gimli;

// issue #74672
// @!has foo/trait.Deref.html '//*[@id="impl-Deref-for-EndianSlice"]//code' 'impl Deref for EndianSlice'
pub use realcore::Deref;

// @has foo/trait.Join.html '//*[@id="impl-Join-for-Foo"]//code' 'impl Join for Foo'
pub use realcore::Join;
