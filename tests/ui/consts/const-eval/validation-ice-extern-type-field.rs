#![feature(extern_types)]

extern {
    type Opaque;
}

struct ThinDst {
    x: u8,
    tail: Opaque,
}

const C1: &ThinDst = unsafe { std::mem::transmute(b"d".as_ptr()) };
//~^ERROR: evaluation of constant value failed

fn main() {}
