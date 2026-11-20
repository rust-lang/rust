// This test ensures that the `doc(fake_variadic)` attribute is correctly handled
// through reexports.

//@ aux-build:reexport-fake_variadic.rs

#![crate_name = "foo"]

extern crate reexport_fake_variadic as dep;

//@ has foo/trait.Foo.html
//@ has - '//section[@id="impl-Foo-for-(T,)"]/h3' 'impl<T> Foo for (T₁, T₂, …, Tₙ)'
pub use dep::Foo;
