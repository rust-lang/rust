// Test that we can handle unsized types with an extern type tail part.
// Regression test for issue #91827.

#![feature(extern_types)]

use std::ptr::addr_of;

extern "C" {
    type Opaque;
}

struct Newtype(Opaque);

struct S {
    i: i32,
    j: i32,
    a: Newtype,
}

const NEWTYPE: () = unsafe {
    let buf = [0i32; 4];
    let x: &Newtype = &*(&buf as *const _ as *const Newtype);

    // Projecting to the newtype works, because it is always at offset 0.
    let field = &x.0;
};

const OFFSET: () = unsafe {
    let buf = [0i32; 4];
    let x: &S = &*(&buf as *const _ as *const S);

    // Accessing sized fields is perfectly fine, even at non-zero offsets.
    let field = &x.i;
    let field = &x.j;

    // This needs to compute the field offset, but we don't know the type's alignment, so this
    // fails.
    let field = &x.a;
    //~^ ERROR: evaluation of constant value failed
    //~| NOTE does not have a known offset
};

fn main() {}
