// aux-build:src-links-external.rs
// build-aux-docs
// ignore-cross-compile
// ignore-tidy-linelength

#![crate_name = "foo"]

extern crate src_links_external;

// @has foo/bar/index.html '//a/@href' '../../src/src_links_external/src-links-external.rs.html#1'
#[doc(inline)]
pub use src_links_external as bar;

// @has foo/bar/struct.Foo.html '//a/@href' '../../src/src_links_external/src-links-external.rs.html#1'
