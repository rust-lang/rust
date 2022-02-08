// Reexport of a structure that derefs to a type with lang item impls having doc links in their
// comments. The doc link points to an associated item, so we check that traits in scope for that
// link are populated.

// aux-build:extern-lang-item-impl-dep.rs

#![no_std]

extern crate extern_lang_item_impl_dep;

pub use extern_lang_item_impl_dep::DerefsToF64;
