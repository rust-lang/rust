// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
#![feature(core_intrinsics)]
#![feature(never_type)]

use std::intrinsics;
use std::mem;

#[derive(Copy, Clone)]
enum Bar {}

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

const BAD_BAD_BAD: Bar = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed
//~| constructing invalid value

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR it is undefined behavior to use this value
//~| constructing invalid value

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR evaluation of constant value failed
//~| constructing invalid value


const READ_NEVER: () = unsafe {
    let mem = [0u32; 8];
    let ptr = mem.as_ptr().cast::<!>();
    let _val = intrinsics::read_via_copy(ptr);
    //~^ ERROR evaluation of constant value failed
    //~| constructing invalid value
};


fn main() {}
