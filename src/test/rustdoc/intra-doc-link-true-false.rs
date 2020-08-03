#![deny(broken_intra_doc_links)]
#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/index.html
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/keyword.true.html"]' 'true'
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/keyword.false.html"]' 'false'

//! A `bool` is either [`true`] or [`false`].
