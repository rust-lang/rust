#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

//@ set OwnerMetadata = '$.index[?(@.name=="OwnerMetadata")].id'
pub struct OwnerMetadata;
//@ set Owner = '$.index[?(@.name=="Owner")].id'
pub struct Owner;

pub fn create() -> Owner::Metadata {
    OwnerMetadata
}
//@ is '$.index[?(@.name=="create")].inner.function.sig.output.qualified_path.name' '"Metadata"'
//@ is '$.index[?(@.name=="create")].inner.function.sig.output.qualified_path.trait' null
//@ is '$.index[?(@.name=="create")].inner.function.sig.output.qualified_path.self_type.resolved_path.id' $Owner

/// impl
impl Owner {
    /// iat
    pub type Metadata = OwnerMetadata;
}
//@ set iat = '$.index[?(@.docs=="iat")].id'
//@ is '$.index[?(@.docs=="impl")].inner.impl.items[*]' $iat
//@ is '$.index[?(@.docs=="iat")].inner.assoc_type.type.resolved_path.id' $OwnerMetadata
