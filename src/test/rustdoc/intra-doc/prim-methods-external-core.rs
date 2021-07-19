// aux-build:my-core.rs
// build-aux-docs
// ignore-cross-compile
// ignore-windows

#![deny(broken_intra_doc_links)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "rlib"]

// @has prim_methods_external_core/index.html
// @has - '//*[@id="main"]//a[@href="../my_core/primitive.char.html"]' 'char'
// @has - '//*[@id="main"]//a[@href="../my_core/primitive.char.html#method.len_utf8"]' 'char::len_utf8'

//! A [`char`] and its [`char::len_utf8`].

extern crate my_core;
