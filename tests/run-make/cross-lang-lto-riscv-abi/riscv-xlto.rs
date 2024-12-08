#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[no_mangle]
pub fn hello() {}
