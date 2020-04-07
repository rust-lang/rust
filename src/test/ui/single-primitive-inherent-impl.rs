// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(lang_items)]
#![no_std]

// OK
#[lang = "str_alloc_impl"]
impl str {}

impl str {
//~^ error: only a single inherent implementation marked with `#[lang = "str_impl"]` is allowed for the `str` primitive
}
