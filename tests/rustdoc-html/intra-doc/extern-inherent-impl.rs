// Reexport of a structure with public inherent impls having doc links in their comments. The doc
// link points to an associated item, so we check that traits in scope for that link are populated.

//@ aux-build:extern-inherent-impl-dep.rs

extern crate extern_inherent_impl_dep;

pub use extern_inherent_impl_dep::PublicStruct;
