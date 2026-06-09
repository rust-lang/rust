//@ build-pass
// regression test for #112051, not in `offset-of-dst` as the issue is in codegen,
// and isn't triggered in the presence of typeck errors

#![feature(extern_types)]

use std::mem::offset_of;

#[repr(C)]
struct Alpha {
    x: u8,
    y: u16,
    z: [u8],
}

trait Trait {}

#[repr(C)]
struct Beta {
    x: u8,
    y: u16,
    z: dyn Trait,
}

unsafe extern "C" {
    type Extern;
}

#[repr(C)]
struct Gamma {
    x: u8,
    y: u16,
    z: Extern,
}

struct S<T: ?Sized> {
    a: u64,
    b: T,
}

fn main() {
    let _ = offset_of!(Alpha, x);
    let _ = offset_of!(Alpha, y);

    let _ = offset_of!(Beta, x);
    let _ = offset_of!(Beta, y);

    let _ = offset_of!(Gamma, x);
    let _ = offset_of!(Gamma, y);

    let _ = offset_of!(S<dyn Trait>, a);
    let _ = offset_of!((u64, dyn Trait), 0);
}
