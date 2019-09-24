// build-pass (FIXME(62277): could be check-pass?)

#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

extern {
    pub static A: u32;
}
