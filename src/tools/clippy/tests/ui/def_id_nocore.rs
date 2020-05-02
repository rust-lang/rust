// ignore-windows
// ignore-macos

#![feature(no_core, lang_items, start)]
#![no_core]

#[link(name = "c")]
extern "C" {}

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
#[lang = "freeze"]
pub unsafe trait Freeze {}

#[lang = "start"]
#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}

pub struct A;

impl A {
    pub fn as_ref(self) -> &'static str {
        "A"
    }
}
