//@ run-pass

// Tests that we can compare various kinds of extern fn signatures.
#![allow(non_camel_case_types)]
#![allow(unpredictable_function_pointer_comparisons)]

// `dbg!()` differentiates these functions to ensure they won't be merged.
extern "C" fn voidret1() { dbg!() }
extern "C" fn voidret2() { dbg!() }

extern "C" fn uintret() -> usize { 22 }

extern "C" fn uintvoidret(_x: usize) {}

extern "C" fn uintuintuintuintret(x: usize, y: usize, z: usize) -> usize { x+y+z }
type uintuintuintuintret = extern "C" fn(usize,usize,usize) -> usize;

pub fn main() {
    assert!(voidret1 as extern "C" fn() == voidret1 as extern "C" fn());
    assert!(voidret1 as extern "C" fn() != voidret2 as extern "C" fn());

    assert!(uintret as extern "C" fn() -> usize == uintret as extern "C" fn() -> usize);

    assert!(uintvoidret as extern "C" fn(usize) == uintvoidret as extern "C" fn(usize));

    assert!(uintuintuintuintret as uintuintuintuintret ==
            uintuintuintuintret as uintuintuintuintret);
}
