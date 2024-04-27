#![feature(extern_types)]

extern "C" {
    type Opaque;
}

struct Newtype(Opaque);

struct S {
    i: i32,
    j: i32,
    a: Newtype,
}

fn main() {
    let buf = [0i32; 4];

    let x: &Newtype = unsafe { &*(&buf as *const _ as *const Newtype) };
    // Projecting to the newtype works, because it is always at offset 0.
    let _field = &x.0;

    let x: &S = unsafe { &*(&buf as *const _ as *const S) };
    // Accessing sized fields is perfectly fine, even at non-zero offsets.
    let _field = &x.i;
    let _field = &x.j;
    // This needs to compute the field offset, but we don't know the type's alignment,
    // so this panics.
    let _field = &x.a; //~ERROR: does not have a known offset
}
