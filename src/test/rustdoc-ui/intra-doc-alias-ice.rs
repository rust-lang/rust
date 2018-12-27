#![deny(intra_doc_link_resolution_failure)]

pub type TypeAlias = usize;

/// [broken cross-reference](TypeAlias::hoge) //~ ERROR
pub fn some_public_item() {}
