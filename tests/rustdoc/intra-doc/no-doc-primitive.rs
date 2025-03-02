// Crate tree without a `rustc_doc_primitive` module for primitive type linked to by a doc link.

#![deny(rustdoc::broken_intra_doc_links)]
#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]
#![rustc_coherence_is_core]
#![crate_type = "rlib"]


//@ has no_doc_primitive/index.html
//! A [`char`] and its [`char::len_utf8`].

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

impl char {
    pub fn len_utf8(self) -> usize {
        42
    }
}
