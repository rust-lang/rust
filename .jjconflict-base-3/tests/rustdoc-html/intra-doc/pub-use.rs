//@ aux-build: intra-link-pub-use.rs
#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "outer"]

extern crate inner;

/// [mod@std::env] [g]
//@ has outer/index.html
//@ has - '//a[@href="{{channel}}/std/env/index.html"]' "std::env"
//@ has - '//a[@href="fn.g.html"]' "g"
pub use f as g;

// Make sure the documentation is actually correct by documenting an inlined re-export
/// [mod@std::env]
//@ has outer/fn.f.html
//@ has - '//a[@href="{{channel}}/std/env/index.html"]' "std::env"
pub use inner::f;
