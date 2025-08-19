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

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "clone"]
pub trait Clone: Sized {}

#[lang = "copy"]
pub trait Copy: Clone {}
