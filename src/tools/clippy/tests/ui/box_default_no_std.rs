#![feature(lang_items, start, libc)]
#![warn(clippy::box_default)]
#![no_std]

pub struct NotBox<T> {
    _value: T,
}

impl<T> NotBox<T> {
    pub fn new(value: T) -> Self {
        Self { _value: value }
    }
}

impl<T: Default> Default for NotBox<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let _p = NotBox::new(isize::default());
    0
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
