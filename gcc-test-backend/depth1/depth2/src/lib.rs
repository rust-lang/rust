#![feature(lang_items, no_core)]
#![no_core]

use mini_core::Sync;

pub struct Struct5 {
    pub field1: &'static [u8],
    pub field2: i32,
}

unsafe impl Sync for Struct5 {}

pub static STRUCT5: Struct5 = Struct5 {
    field1: b"depth2",
    field2: 12,
};

pub static STRUCT6: Struct5 = Struct5 {
    field1: b"depth2",
    field2: 12,
};
