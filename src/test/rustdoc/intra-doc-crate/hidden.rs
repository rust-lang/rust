// aux-build:hidden.rs
// build-aux-docs
#![deny(intra_doc_link_resolution_failure)]

// tests https://github.com/rust-lang/rust/issues/73363

extern crate hidden_dep;

// @has 'hidden/struct.Ready.html' '//a/@href' '../hidden/fn.ready.html'
pub use hidden_dep::future::{ready, Ready};
