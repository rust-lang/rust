// aux-build:my-core.rs
// ignore-cross-compile

#![deny(intra_doc_link_resolution_failure)]
#![feature(no_core, lang_items)]
#![no_core]

//! A [`char`] and its [`char::len_utf8`].

extern crate my_core;
