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

struct Delta<T: ?Sized> {
    x: u8,
    y: u16,
    z: T,
}

fn main() {
    offset_of!(Alpha, z); //~ ERROR the size for values of type
    offset_of!(Beta, z); //~ ERROR the size for values of type
    offset_of!(Gamma, z); //~ ERROR the size for values of type
    offset_of!((u8, dyn Trait), 0); // ok
    offset_of!((u8, dyn Trait), 1); //~ ERROR the size for values of type
}

fn delta() {
    offset_of!(Delta<Alpha>, z); //~ ERROR the size for values of type
    offset_of!(Delta<Extern>, z); //~ ERROR the size for values of type
    offset_of!(Delta<dyn Trait>, z); //~ ERROR the size for values of type
}

fn generic_with_maybe_sized<T: ?Sized>() -> usize {
    offset_of!(Delta<T>, z) //~ ERROR the size for values of type
}
