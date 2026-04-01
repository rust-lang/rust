// Ensure that no warning is emitted if the disambiguator is used for type alias.
// Regression test for <https://github.com/rust-lang/rust/issues/146855>.

#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo;

#[allow(non_camel_case_types)]
pub type f32 = Foo;

/// This function returns [`f32`].
//~^ ERROR: `f32` is both a type alias and a primitive type
//~| HELP: to link to the type alias, prefix with `tyalias@`
//~| HELP: to link to the primitive type, prefix with `prim@`
pub fn my_fn() -> f32 {}

/// This function returns [type@f32].
//~^ ERROR: `f32` is both a type alias and a primitive type
//~| HELP: to link to the type alias, prefix with `tyalias@`
//~| HELP: to link to the primitive type, prefix with `prim@`
pub fn my_fn2() -> f32 {}
