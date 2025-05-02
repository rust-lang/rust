#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "rlib"]

#[lang = "sized"]
trait Sized {}

pub fn foo() {}
