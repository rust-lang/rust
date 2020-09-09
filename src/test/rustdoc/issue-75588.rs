// aux-build:all_unstable.rs

// Ensure unstably exported traits have their Implementors sections.

#![crate_name = "foo"]
#![feature(extremely_unstable_foo)]

extern crate unstabled;

// @has foo/trait.Join.html '//*[@id="impl-Join-for-Foo"]//code' 'impl Join for Foo'
pub use unstabled::Join;
