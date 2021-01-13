// run-pass
// Tests that we can compare various kinds of extern fn signatures.
#![allow(non_camel_case_types)]

// `dbg!()` differentiates these functions to ensure they won't be merged.
extern fn voidret1() { dbg!() }
extern fn voidret2() { dbg!() }

extern fn uintret() -> usize { 22 }

extern fn uintvoidret(_x: usize) {}

extern fn uintuintuintuintret(x: usize, y: usize, z: usize) -> usize { x+y+z }
type uintuintuintuintret = extern fn(usize,usize,usize) -> usize;

pub fn main() {
    assert!(voidret1 as extern fn() == voidret1 as extern fn());
    assert!(voidret1 as extern fn() != voidret2 as extern fn());

    assert!(uintret as extern fn() -> usize == uintret as extern fn() -> usize);

    assert!(uintvoidret as extern fn(usize) == uintvoidret as extern fn(usize));

    assert!(uintuintuintuintret as uintuintuintuintret ==
            uintuintuintuintret as uintuintuintuintret);
}
