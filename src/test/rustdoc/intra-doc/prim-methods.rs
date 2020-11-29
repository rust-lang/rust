#![deny(broken_intra_doc_links)]

// ignore-tidy-linelength

// @has prim_methods/index.html
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/primitive.char.html"]' 'char'
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/primitive.char.html#method.len_utf8"]' 'char::len_utf8'

//! A [`char`] and its [`char::len_utf8`].
