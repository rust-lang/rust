#![feature(offset_of, extern_types)]

use std::mem::offset_of;

struct Alpha {
    x: u8,
    y: u16,
    z: [u8],
}

trait Trait {}

struct Beta {
    x: u8,
    y: u16,
    z: dyn Trait,
}

extern {
    type Extern;
}

struct Gamma {
    x: u8,
    y: u16,
    z: Extern,
}

fn main() {
    offset_of!(Alpha, z); //~ ERROR the size for values of type
    offset_of!(Beta, z); //~ ERROR the size for values of type
    offset_of!(Gamma, z); //~ ERROR the size for values of type
}
