#![feature(no_core)]
#![no_core]

// @set struct = "$.index[*][?(@.name=='Struct')].id"
pub struct Struct;
// @set trait = "$.index[*][?(@.name=='Trait')].id"
pub trait Trait {}
// @set impl = "$.index[*][?(@.docs=='impl')].id"
/// impl
impl Trait for Struct {}

// @is "$.index[*][?(@.name=='Struct')].inner.impls[*]" $impl
// @is "$.index[*][?(@.name=='Trait')].inner.implementations[*]" $impl
// @is "$.index[*][?(@.docs=='impl')].inner.trait.id" $trait
// @is "$.index[*][?(@.docs=='impl')].inner.for.inner.id" $struct
