#![deny(intra_doc_link_resolution_failure)]

/// [std::env] [g]
// @has intra_link_pub_use/index.html '//a[@href="https://doc.rust-lang.org/nightly/std/env/fn.var.html"]' "std::env"
// @has - '//a[@href="../intra_link_pub_use/fn.f.html"]' "g"
pub use f as g;

pub fn f() {}
