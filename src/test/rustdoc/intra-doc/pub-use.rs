// aux-build: intra-link-pub-use.rs
#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "outer"]

extern crate inner;

/// [mod@std::env] [g]

// FIXME: This can't be tested because rustdoc doesn't show documentation on pub re-exports.
// Until then, comment out the `htmldocck` test.
// This test still does something; namely check that no incorrect errors are emitted when
// documenting the re-export.

// @has outer/index.html
// @ has - '//a[@href="{{channel}}/std/env/fn.var.html"]' "std::env"
// @ has - '//a[@href="fn.f.html"]' "g"
pub use f as g;

// FIXME: same as above
/// [std::env]
extern crate self as _;

// Make sure the documentation is actually correct by documenting an inlined re-export
/// [mod@std::env]
// @has outer/fn.f.html
// @has - '//a[@href="{{channel}}/std/env/index.html"]' "std::env"
pub use inner::f;
