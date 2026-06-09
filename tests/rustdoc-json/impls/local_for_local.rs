//@ set struct = "$.index[?(@.name=='Struct')].id"
pub struct Struct;
//@ set trait = "$.index[?(@.name=='Trait')].id"
pub trait Trait {}
//@ set impl = "$.index[?(@.docs=='impl')].id"
/// impl
impl Trait for Struct {}

//@ has "$.index[?(@.name=='Struct')].inner.struct.impls[*]" $impl
//@ is "$.index[?(@.name=='Trait')].inner.trait.implementations[*]" $impl
//@ is "$.index[?(@.docs=='impl')].inner.impl.trait.id" $trait
//@ is "$.index[?(@.docs=='impl')].inner.impl.for.resolved_path.id" $struct
