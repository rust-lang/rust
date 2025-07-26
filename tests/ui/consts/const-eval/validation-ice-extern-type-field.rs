#![feature(extern_types)]

extern "C" {
    type Opaque;
}

struct ThinDst {
    x: u8,
    tail: Opaque,
}

const C1: &ThinDst = unsafe { std::mem::transmute(b"d".as_ptr()) };
//~^ERROR: `extern type` field does not have a known offset

fn main() {}
