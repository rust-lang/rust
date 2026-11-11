// Regression test for <https://github.com/rust-lang/rust/issues/105025>
//@ aux-build: enum_variant_in_trait_method.rs

extern crate enum_variant_in_trait_method;

pub struct Local;

/// local impl
impl enum_variant_in_trait_method::Trait for Local {}

//@ !has "$.index[?(@.name == 'Trait')]"
//@ !has "$.index[?(@.name == 'method')]"
//@ count "$.index[?(@.docs == 'local impl')].inner.items[*]" 0
