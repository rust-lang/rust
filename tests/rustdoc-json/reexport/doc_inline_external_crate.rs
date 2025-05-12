// Regression Test for https://github.com/rust-lang/rust/issues/110138
//@ aux-build: enum_with_discriminant.rs

#[doc(inline)]
pub extern crate enum_with_discriminant;

//@ !has '$.index[?(@.docs == "Should not be inlined")]'
//@ has '$.index[?(@.name == "enum_with_discriminant")].inner.extern_crate'
//@ set enum_with_discriminant = '$.index[?(@.name == "enum_with_discriminant")].id'
//@ is '$.index[?(@.name == "doc_inline_external_crate")].inner.module.items[*]' $enum_with_discriminant
