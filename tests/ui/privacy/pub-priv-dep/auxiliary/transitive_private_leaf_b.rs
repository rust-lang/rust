//@ aux-crate:transitive_private_leaf_c=transitive_private_leaf_c.rs

extern crate transitive_private_leaf_c;

pub use transitive_private_leaf_c::{CType, DType};

pub struct BType;
