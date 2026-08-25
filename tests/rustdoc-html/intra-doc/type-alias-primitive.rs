// Ensure that no warning is emitted if the disambiguator is used for type alias.
// Regression test for <https://github.com/rust-lang/rust/issues/146855>.

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo;

#[allow(non_camel_case_types)]
pub type f32 = Foo;

/// This function returns [`tyalias@f32`] and not [bla][`prim@f32`].
//@ has 'foo/fn.my_fn.html'
//@ has - '//a[@href="type.f32.html"]' "f32"
//@ has - '//a[@href="{{channel}}/std/primitive.f32.html"]' "bla"
pub fn my_fn() -> f32 { 0. }

/// This function returns [`typealias@f32`].
//@ has 'foo/fn.my_other_fn.html'
//@ has - '//a[@href="type.f32.html"]' "f32"
pub fn my_other_fn() -> f32 { 0. }
