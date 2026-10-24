// Ensure that no warning is emitted if the disambiguator is used for type alias.
// Regression test for <https://github.com/rust-lang/rust/issues/146855>.

//@ check-pass

#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo;

#[allow(non_camel_case_types)]
pub type f32 = Foo;

/// This function returns [`tyalias@f32`] and not [`prim@f32`].
pub fn my_fn() -> f32 {}
