// run-pass
#![allow(unused_imports)]
#![feature(lang_items, start)]
#![no_std]

extern crate std as other;

#[macro_use] extern crate alloc;

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    for _ in [1,2,3].iter() { }
    0
}
