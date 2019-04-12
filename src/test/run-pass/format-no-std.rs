// ignore-emscripten no no_std executables

#![feature(lang_items, start)]
#![no_std]

extern crate std as other;

#[macro_use] extern crate alloc;

use alloc::string::ToString;

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    let s = format!("{}", 1_isize);
    assert_eq!(s, "1".to_string());

    let s = format!("test");
    assert_eq!(s, "test".to_string());

    let s = format!("{test}", test=3_isize);
    assert_eq!(s, "3".to_string());

    let s = format!("hello {}", "world");
    assert_eq!(s, "hello world".to_string());

    0
}
