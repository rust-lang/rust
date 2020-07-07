#![deny(intra_doc_link_resolution_failure)]
#![feature(no_core, lang_items)]
#![no_core]

//! A [`char`] and its [`char::len_utf8`].

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
