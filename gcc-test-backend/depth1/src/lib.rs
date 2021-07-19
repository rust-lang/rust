#![feature(no_core)]
#![no_core]

use mini_core::Sync;
use mini_core::libc::printf;

pub use depth2::STRUCT5;

pub struct Struct4 {
    pub field1: &'static [u8],
    pub field2: i32,
}

unsafe impl Sync for Struct4 {}

pub static STRUCT4: Struct4 = Struct4 {
    field1: b"depth1",
    field2: 12,
};

pub fn print_ptr() {
    use depth2::STRUCT6;
    unsafe { printf("ptr: %p\n\0" as *const str as *const i8, STRUCT6.field1) };
}
