// aux-build: foreign_struct.rs
extern crate foreign_struct;

/// ForeignStruct id hack
pub use foreign_struct::ForeignStruct as _;
// @set ForeignStruct = "$.index[*][?(@.docs=='ForeignStruct id hack')].inner.id"

pub trait LocalTrait {}
// @set LocalTrait = "$.index[*][?(@.name=='LocalTrait')].id"

/// local for foreign
impl LocalTrait for foreign_struct::ForeignStruct {}

// @set impl = "$.index[*][?(@.docs=='local for foreign')].id"
// @is "$.index[*][?(@.docs=='local for foreign')].inner.trait.id" $LocalTrait
// @is "$.index[*][?(@.docs=='local for foreign')].inner.for.inner.id" $ForeignStruct

// @is "$.index[*][?(@.name=='LocalTrait')].inner.implementations[*]" $impl
