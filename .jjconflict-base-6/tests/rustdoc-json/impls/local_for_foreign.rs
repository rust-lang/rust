//@ aux-build: foreign_struct.rs
extern crate foreign_struct;

/// ForeignStruct id hack
pub use foreign_struct::ForeignStruct as _;
//@ set ForeignStruct = "$.index[?(@.docs=='ForeignStruct id hack')].inner.use.id"

pub trait LocalTrait {}
//@ set LocalTrait = "$.index[?(@.name=='LocalTrait')].id"

/// local for foreign
impl LocalTrait for foreign_struct::ForeignStruct {}

//@ set impl = "$.index[?(@.docs=='local for foreign')].id"
//@ is "$.index[?(@.docs=='local for foreign')].inner.impl.trait.id" $LocalTrait
//@ is "$.index[?(@.docs=='local for foreign')].inner.impl.for.resolved_path.id" $ForeignStruct

//@ is "$.index[?(@.name=='LocalTrait')].inner.trait.implementations[*]" $impl
