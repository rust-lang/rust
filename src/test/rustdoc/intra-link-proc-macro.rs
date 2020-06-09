// aux-build:intra-link-proc-macro-macro.rs
// build-aux-docs
// @has intra_link_proc_macro/index.html
#![deny(intra_doc_link_resolution_failure)]

extern crate intra_link_proc_macro_macro;


pub use intra_link_proc_macro_macro::{DeriveA, attr_a};
use intra_link_proc_macro_macro::{DeriveB, attr_b};

// @has - '//a/@href' '../intra_link_proc_macro/derive.DeriveA.html'
// @has - '//a/@href' '../intra_link_proc_macro/attr.attr_a.html'
// @has - '//a/@href' '../intra_link_proc_macro_macro/derive.DeriveB.html'
// @has - '//a/@href' '../intra_link_proc_macro_macro/attr.attr_b.html'
/// Link to [DeriveA], [attr_a], [DeriveB], [attr_b]
pub struct Foo;
