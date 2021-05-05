#![deny(broken_intra_doc_links)]


// @has prim_methods/index.html
// @has - '//*[@id="main"]//a[@href="{{channel}}/std/primitive.char.html"]' 'char'
// @has - '//*[@id="main"]//a[@href="{{channel}}/std/primitive.char.html#method.len_utf8"]' 'char::len_utf8'

//! A [`char`] and its [`char::len_utf8`].
