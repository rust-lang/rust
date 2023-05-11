// aux-build: enum_with_discriminant.rs

extern crate enum_with_discriminant;

#[doc(inline)]
pub use enum_with_discriminant::*;

// @!has '$.index[*][?(@.docs == "Should not be inlined")]'
// @set use = '$.index[*][?(@.inner.name == "enum_with_discriminant")].id'
// @is '$.index[*][?(@.name == "extern_crate_glob")].inner.items[*]' $use
