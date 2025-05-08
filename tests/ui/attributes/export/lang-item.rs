#![feature(no_core, lang_items, export_stable)]
#![allow(incomplete_features)]
#![crate_type = "sdylib"]
#![no_core]

#[lang = "sized"]
//~^ ERROR lang items are not allowed in stable dylibs
trait Sized {}
