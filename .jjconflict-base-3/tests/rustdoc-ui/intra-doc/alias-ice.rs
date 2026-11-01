#![deny(rustdoc::broken_intra_doc_links)]

pub type TypeAlias = usize;

/// [broken cross-reference](TypeAlias::hoge) //~ ERROR
pub fn some_public_item() {}
