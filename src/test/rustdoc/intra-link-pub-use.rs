#![deny(broken_intra_doc_links)]

/// [std::env] [g]
// FIXME: This can't be tested because rustdoc doesn't show documentation on pub re-exports.
// Until then, comment out the `htmldocck` test.
// This test still does something; namely check that no incorrect errors are emitted when
// documenting the re-export.

// @has intra_link_pub_use/index.html
// @ has - '//a[@href="https://doc.rust-lang.org/nightly/std/env/fn.var.html"]' "std::env"
// @ has - '//a[@href="../intra_link_pub_use/fn.f.html"]' "g"
pub use f as g;

/// [std::env]
extern crate self as _;

pub fn f() {}
