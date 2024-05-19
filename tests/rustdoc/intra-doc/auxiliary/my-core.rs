#![feature(no_core, lang_items, rustdoc_internals, rustc_attrs)]
#![no_core]
#![rustc_coherence_is_core]
#![crate_type="rlib"]

#[rustc_doc_primitive = "char"]
/// Some char docs
mod char {}

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
