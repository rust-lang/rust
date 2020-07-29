#![deny(intra_doc_resolution_failures)]

pub type TypeAlias = usize;

/// [broken cross-reference](TypeAlias::hoge) //~ ERROR
pub fn some_public_item() {}
