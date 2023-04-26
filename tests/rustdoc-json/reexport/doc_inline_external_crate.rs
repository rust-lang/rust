// Regression Test for https://github.com/rust-lang/rust/issues/110138
// aux-build: enum_with_discriminant.rs

#[doc(inline)]
pub extern crate enum_with_discriminant;

// @!has '$.index[*][?(@.docs == "Should not be inlined")]'
// @is '$.index[*][?(@.name == "enum_with_discriminant")].kind' '"extern_crate"'
// @set enum_with_discriminant = '$.index[*][?(@.name == "enum_with_discriminant")].id'
// @is '$.index[*][?(@.name == "doc_inline_external_crate")].inner.items[*]' $enum_with_discriminant
