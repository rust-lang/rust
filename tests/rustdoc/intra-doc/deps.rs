// Checks that links to crates are correctly generated and only existing crates
// have a link generated.
// Regression test for <https://github.com/rust-lang/rust/issues/137857>.

//@ compile-flags: --document-private-items -Z unstable-options
//@ compile-flags: --extern-html-root-url=empty=https://empty.example/
// This one is to ensure that we don't link to any item we see which has
// an external html root URL unless it actually exists.
//@ compile-flags: --extern-html-root-url=non_existant=https://non-existant.example/
//@ aux-build: empty.rs

#![crate_name = "foo"]
#![expect(rustdoc::broken_intra_doc_links)]

//@ has 'foo/index.html'
//@ has - '//a[@href="https://empty.example/empty/index.html"]' 'empty'
// There should only be one intra doc links, we should not link `non_existant`.
//@ count - '//*[@class="docblock"]//a' 1
//! [`empty`]
//!
//! [`non_existant`]

extern crate empty;
