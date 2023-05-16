// ignore-tidy-linelength
#![feature(inherent_associated_types)]
#![feature(no_core)]
#![allow(incomplete_features)]
#![no_core]

// @set OwnerMetadata = '$.index[*][?(@.name=="OwnerMetadata")].id'
pub struct OwnerMetadata;
// @set Owner = '$.index[*][?(@.name=="Owner")].id'
pub struct Owner;

pub fn create() -> Owner::Metadata {
    OwnerMetadata
}
// @is '$.index[*][?(@.name=="create")].inner.decl.output.kind' '"qualified_path"'
// @is '$.index[*][?(@.name=="create")].inner.decl.output.inner.name' '"Metadata"'
// @is '$.index[*][?(@.name=="create")].inner.decl.output.inner.trait' null
// @is '$.index[*][?(@.name=="create")].inner.decl.output.inner.self_type.kind' '"resolved_path"'
// @is '$.index[*][?(@.name=="create")].inner.decl.output.inner.self_type.inner.id' $Owner

/// impl
impl Owner {
    /// iat
    pub type Metadata = OwnerMetadata;
}
// @set iat = '$.index[*][?(@.docs=="iat")].id'
// @is '$.index[*][?(@.docs=="impl")].inner.items[*]' $iat
// @is '$.index[*][?(@.docs=="iat")].kind' '"assoc_type"'
// @is '$.index[*][?(@.docs=="iat")].inner.default.inner.id' $OwnerMetadata
