#![allow(unused)]
#![deny(improper_ctypes)]

#[repr(C)]
union U {
    a: u8,
}

union W {
    a: u8,
}

extern "C" {
    static FOREIGN1: U; // OK
    static FOREIGN2: W; //~ ERROR union has unspecified layout
}

fn main() {}
