#![deny(intra_doc_link_resolution_failure)]

// ignore-tidy-linelength

// @has intra_link_prim_methods/index.html
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/primitive.char.html"]' 'char'
// @has - '//*[@id="main"]//a[@href="https://doc.rust-lang.org/nightly/std/primitive.char.html#method.len_utf8"]' 'char::len_utf8'

//! A [`char`] and its [`char::len_utf8`].
