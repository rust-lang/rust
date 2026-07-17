//@ aux-crate:priv:transitive_private_leaf_d=transitive_private_leaf_d.rs
//@ compile-flags: -Zunstable-options

#![allow(exported_private_dependencies)]

extern crate transitive_private_leaf_d;

pub use transitive_private_leaf_d::DType;

pub struct CType;
