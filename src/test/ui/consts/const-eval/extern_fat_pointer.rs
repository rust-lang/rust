// check-pass

#![feature(extern_types)]

extern "C" {
    type Opaque;
}

const FOO: *const u8 = &42 as *const _ as *const Opaque as *const u8;

fn main() {
    let _foo = FOO;
}
