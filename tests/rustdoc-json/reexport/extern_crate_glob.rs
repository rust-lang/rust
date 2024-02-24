//@ aux-build: enum_with_discriminant.rs

extern crate enum_with_discriminant;

#[doc(inline)]
pub use enum_with_discriminant::*;

// @!has '$.index[*][?(@.docs == "Should not be inlined")]'
// @is '$.index[*][?(@.inner.import)].inner.import.name' \"enum_with_discriminant\"
// @set use = '$.index[*][?(@.inner.import)].id'
// @is '$.index[*][?(@.name == "extern_crate_glob")].inner.module.items[*]' $use
