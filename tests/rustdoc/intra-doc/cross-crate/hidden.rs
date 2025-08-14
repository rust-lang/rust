//@ aux-build:hidden.rs
//@ build-aux-docs
#![deny(rustdoc::broken_intra_doc_links)]

// tests https://github.com/rust-lang/rust/issues/73363

extern crate hidden_dep;

//@ has 'hidden/struct.Ready.html' '//a/@href' 'fn.ready.html'
pub use hidden_dep::future::{ready, Ready};
