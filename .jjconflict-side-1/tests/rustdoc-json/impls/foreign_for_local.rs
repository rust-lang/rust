//@ aux-build: foreign_trait.rs
extern crate foreign_trait;

/// ForeignTrait id hack
pub use foreign_trait::ForeignTrait as _;
//@ set ForeignTrait = "$.index[?(@.docs=='ForeignTrait id hack')].inner.use.id"

pub struct LocalStruct;
//@ set LocalStruct = "$.index[?(@.name=='LocalStruct')].id"

/// foreign for local
impl foreign_trait::ForeignTrait for LocalStruct {}

//@ set impl = "$.index[?(@.docs=='foreign for local')].id"
//@ is "$.index[?(@.docs=='foreign for local')].inner.impl.for.resolved_path.id" $LocalStruct
//@ is "$.index[?(@.docs=='foreign for local')].inner.impl.trait.id" $ForeignTrait

//@ has "$.index[?(@.name=='LocalStruct')].inner.struct.impls[*]" $impl
