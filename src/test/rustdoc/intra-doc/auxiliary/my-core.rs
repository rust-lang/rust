#![feature(no_core, lang_items, rustdoc_internals)]
#![no_core]
#![crate_type="rlib"]

#[doc(primitive = "char")]
/// Some char docs
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
