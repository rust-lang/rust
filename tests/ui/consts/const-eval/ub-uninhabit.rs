// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE

#![feature(core_intrinsics)]
#![feature(never_type)]

use std::{intrinsics, mem};

#[derive(Copy, Clone)]
enum Bar {}

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

const BAD_BAD_BAD: Bar = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR value of zero-variant enum `Bar`

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR reference pointing to uninhabited type `Bar`

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR value of zero-variant enum `Bar`

const READ_NEVER: () = unsafe {
    let mem = [0u32; 8];
    let ptr = mem.as_ptr().cast::<!>();
    let _val = intrinsics::read_via_copy(ptr);
    //~^ ERROR value of the never type
};

const BAD_NESTED_REF: &&! = unsafe { mem::transmute(&&0) };
//~^ ERROR reference pointing to uninhabited type `&!`

const BAD_NESTED_UNSIZED_REF: &&(!, [i32]) = unsafe { mem::transmute(&(&[0] as &[i32])) };
//~^ ERROR reference pointing to uninhabited type `&(!, [i32])`

const BAD_UNINHABITED_SLICE: &[!] = unsafe { mem::transmute(&[()] as &[()]) };
//~^ ERROR value of the never type

// This is an interesting type since it looks somewhat recursive even though it is not.
struct Wrap<T>(T);
const WRAPPED_TWICE: Wrap<Wrap<!>> = unsafe { mem::transmute(()) };
//~^ ERROR value of the never type
const WRAPPED_TWICE_REF: &Wrap<Wrap<!>> = unsafe { mem::transmute(&()) };
//~^ ERROR reference pointing to uninhabited type `Wrap<Wrap<!>>`

fn main() {}
