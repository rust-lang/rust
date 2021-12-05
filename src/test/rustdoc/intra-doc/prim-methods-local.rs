#![deny(rustdoc::broken_intra_doc_links)]
#![feature(no_core, lang_items, rustdoc_internals)]
#![no_core]
#![crate_type = "rlib"]

// @has prim_methods_local/index.html
// @has - '//*[@id="main-content"]//a[@href="primitive.char.html"]' 'char'
// @has - '//*[@id="main-content"]//a[@href="primitive.char.html#method.len_utf8"]' 'char::len_utf8'

//! A [prim@`char`] and its [`char::len_utf8`].

#[doc(primitive = "char")]
mod char {}

#[lang = "char"]
impl char {
    pub fn len_utf8(self) -> usize {
        42
    }
}

#[lang = "sized"]
pub trait Sized {}

#[lang = "clone"]
pub trait Clone: Sized {}

#[lang = "copy"]
pub trait Copy: Clone {}
