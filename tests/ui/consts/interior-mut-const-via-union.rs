// Check that constants with interior mutability inside unions are rejected
// during validation.
//
//@ build-fail
//@ stderr-per-bitwidth
#![feature(const_mut_refs)]

use std::cell::Cell;
use std::mem::ManuallyDrop;

#[repr(C)]
struct S {
    x: u32,
    y: E,
}

#[repr(u32)]
enum E {
    A,
    B(U)
}

union U {
    cell: ManuallyDrop<Cell<u32>>,
}

const C: S = {
    let mut s = S { x: 0, y: E::A };
    let p = &mut s.x as *mut u32;
    // Change enum tag to E::B. Now there's interior mutability here.
    unsafe { *p.add(1) = 1 };
    s
};

fn main() { //~ ERROR it is undefined behavior to use this value
    // FIXME the span here is wrong, sould be pointing at the line below, not above.
    let _: &'static _ = &C;
}
