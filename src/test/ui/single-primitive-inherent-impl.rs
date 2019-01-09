// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(lang_items)]
#![no_std]

// OK
#[lang = "str_alloc"]
impl str {}

impl str {
//~^ error: only a single inherent implementation marked with `#[lang = "str"]` is allowed for the `str` primitive
}
