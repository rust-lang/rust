//@ run-pass
//@ compile-flags: -C debug-assertions

// This is a regression test for https://github.com/rust-lang/rust/issues/159433

use std::{mem, num::NonZeroI32};

type Full = Result<(), NonZeroI32>;

#[repr(C)]
struct Payload {
    pad: u8,
    value: NonZeroI32,
}

enum Partial {
    #[allow(unused)]
    Value(Payload),
    #[allow(unused)]
    Empty,
}

fn main() {
    let _ = unsafe { std::mem::transmute::<Full, Full>(Err(NonZeroI32::MIN)) };

    let value = Partial::Value(Payload {
        pad: 0u8,
        value: NonZeroI32::MIN,
    });
    let _ = unsafe { mem::transmute::<Partial, Partial>(value) };
}
