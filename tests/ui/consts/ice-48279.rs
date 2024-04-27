//@ run-pass
#![allow(dead_code)]
#![allow(unused_unsafe)]

// https://github.com/rust-lang/rust/issues/48279

#[derive(PartialEq, Eq)]
pub struct NonZeroU32 {
    value: u32
}

impl NonZeroU32 {
    const unsafe fn new_unchecked(value: u32) -> Self {
        NonZeroU32 { value }
    }
}

// pub const FOO_ATOM: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(7) };
pub const FOO_ATOM: NonZeroU32 = unsafe { NonZeroU32 { value: 7 } };

fn main() {
    match None {
        Some(FOO_ATOM) => {}
        _ => {}
    }
}
