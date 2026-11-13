// Test for issue #108501.
// Module parent scope doesn't hijack import's parent scope for the import's doc links.

//@ check-pass
//@ aux-build: inner-crate-doc.rs
//@ compile-flags: --extern inner_crate_doc
//@ edition: 2018

/// Import doc comment [inner_crate_doc]
#[doc(inline)]
pub use inner_crate_doc;
