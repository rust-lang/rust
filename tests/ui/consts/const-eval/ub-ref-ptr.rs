// ignore-tidy-linelength
// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |__ |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
//@ dont-require-annotations: NOTE
//@ normalize-stderr: "0x[0-9](\.\.|\])" -> "0x%$1"
#![feature(rustc_attrs)]
#![allow(invalid_value)]

use std::mem;

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

const UNALIGNED: &u16 = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR constructing invalid value: encountered an unaligned reference (required 2 byte alignment but found 1)

const UNALIGNED_BOX: Box<u16> = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR constructing invalid value: encountered an unaligned box (required 2 byte alignment but found 1)

const NULL: &u16 = unsafe { mem::transmute(0usize) };
//~^ ERROR invalid value

const NULL_BOX: Box<u16> = unsafe { mem::transmute(0usize) };
//~^ ERROR invalid value

const MAYBE_NULL_BOX: Box<()> = unsafe { mem::transmute({
//~^ ERROR maybe-null
    let ref_ = &0u8;
    (ref_ as *const u8).wrapping_add(10)
}) };

// It is very important that we reject this: We do promote `&(4 * REF_AS_USIZE)`,
// but that would fail to compile; so we ended up breaking user code that would
// have worked fine had we not promoted.
const REF_AS_USIZE: usize = unsafe { mem::transmute(&0) };
//~^ ERROR unable to turn pointer into integer

const REF_AS_USIZE_SLICE: &[usize] = &[unsafe { mem::transmute(&0) }];
//~^ ERROR unable to turn pointer into integer

const REF_AS_USIZE_BOX_SLICE: Box<[usize]> = unsafe { mem::transmute::<&[usize], _>(&[mem::transmute(&0)]) };
//~^ ERROR unable to turn pointer into integer

const USIZE_AS_REF: &'static u8 = unsafe { mem::transmute(1337usize) };
//~^ ERROR invalid value

const USIZE_AS_BOX: Box<u8> = unsafe { mem::transmute(1337usize) };
//~^ ERROR invalid value

const UNINIT_PTR: *const i32 = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR uninitialized

const NULL_FN_PTR: fn() = unsafe { mem::transmute(0usize) };
//~^ ERROR invalid value
const UNINIT_FN_PTR: fn() = unsafe { MaybeUninit { uninit: () }.init };
//~^ ERROR uninitialized
const DANGLING_FN_PTR: fn() = unsafe { mem::transmute(13usize) };
//~^ ERROR invalid value
const DATA_FN_PTR: fn() = unsafe { mem::transmute(&13) };
//~^ ERROR invalid value
const MAYBE_NULL_FN_PTR: fn() = unsafe { mem::transmute({
//~^ ERROR invalid value
    fn fun() {}
    let ptr = fun as fn();
    (ptr as *const u8).wrapping_add(10)
}) };

const UNALIGNED_READ: () = unsafe {
    let x = &[0u8; 4];
    let ptr = x.as_ptr().cast::<u32>();
    ptr.read(); //~ ERROR accessing memory
};

// Check the general case of a pointer value not falling into the scalar valid range.
#[rustc_layout_scalar_valid_range_start(1000)]
pub struct High {
    pointer: *const (),
}
static S: u32 = 0; // just a static to construct a pointer with unknown absolute address
const INVALID_VALUE_PTR: High = unsafe { mem::transmute(&S) };
//~^ ERROR invalid value


fn main() {}
