//@ aux-crate:transitive_private_leaf_b=transitive_private_leaf_b.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

// A public dependency chain with a private leaf:
//
//     this crate --(public)--> B --(public)--> C --(PRIVATE)--> D
//
// D is indirect, so only the public first hop through B determines its visibility here.

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate transitive_private_leaf_b;

pub fn exposes_b() -> transitive_private_leaf_b::BType {
    transitive_private_leaf_b::BType
}

pub fn exposes_c() -> transitive_private_leaf_b::CType {
    transitive_private_leaf_b::CType
}

pub fn exposes_d() -> transitive_private_leaf_b::DType {
    transitive_private_leaf_b::DType
}
