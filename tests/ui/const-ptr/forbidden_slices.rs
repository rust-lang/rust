// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr-test: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

#![feature(
    slice_from_ptr_range,
    const_slice_from_ptr_range,
)]
use std::{
    mem::{size_of, MaybeUninit},
    ptr,
    slice::{from_ptr_range, from_raw_parts},
};

// Null is never valid for references
pub static S0: &[u32] = unsafe { from_raw_parts(ptr::null(), 0) };
//~^ ERROR: it is undefined behavior to use this value
pub static S1: &[()] = unsafe { from_raw_parts(ptr::null(), 0) };
//~^ ERROR: it is undefined behavior to use this value

// Out of bounds
pub static S2: &[u32] = unsafe { from_raw_parts(&D0, 2) };
//~^ ERROR: it is undefined behavior to use this value

// Reading uninitialized  data
pub static S4: &[u8] = unsafe { from_raw_parts((&D1) as *const _ as _, 1) }; //~ ERROR: it is undefined behavior to use this value
// Reinterpret pointers as integers (UB in CTFE.)
pub static S5: &[u8] = unsafe { from_raw_parts((&D3) as *const _ as _, size_of::<&u32>()) }; //~ ERROR: it is undefined behavior to use this value
// Layout mismatch
pub static S6: &[bool] = unsafe { from_raw_parts((&D0) as *const _ as _, 4) }; //~ ERROR: it is undefined behavior to use this value

// Reading padding is not ok
pub static S7: &[u16] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = (&D2 as *const Struct as *const u16).add(1);

    from_raw_parts(ptr, 4)
};

// Unaligned read
pub static S8: &[u64] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = (&D4 as *const [u32; 2] as *const u32).byte_add(1).cast::<u64>();

    from_raw_parts(ptr, 1)
};

pub static R0: &[u32] = unsafe { from_ptr_range(ptr::null()..ptr::null()) };
//~^ ERROR it is undefined behavior to use this value
pub static R1: &[()] = unsafe { from_ptr_range(ptr::null()..ptr::null()) }; // errors inside libcore
pub static R2: &[u32] = unsafe {
    let ptr = &D0 as *const u32;
    from_ptr_range(ptr..ptr.add(2)) // errors inside libcore
};
pub static R4: &[u8] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = (&D1) as *const MaybeUninit<&u32> as *const u8;
    from_ptr_range(ptr..ptr.add(1))
};
pub static R5: &[u8] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = &D3 as *const &u32;
    from_ptr_range(ptr.cast()..ptr.add(1).cast())
};
pub static R6: &[bool] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = &D0 as *const u32 as *const bool;
    from_ptr_range(ptr..ptr.add(4))
};
pub static R7: &[u16] = unsafe {
    //~^ ERROR: it is undefined behavior to use this value
    let ptr = (&D2 as *const Struct as *const u16).byte_add(1);
    from_ptr_range(ptr..ptr.add(4))
};
pub static R8: &[u64] = unsafe {
    let ptr = (&D4 as *const [u32; 2] as *const u32).byte_add(1).cast::<u64>();
    from_ptr_range(ptr..ptr.add(1)) //~ inside `R8`
};

// This is sneaky: &D0 and &D0 point to different objects
// (even if at runtime they have the same address)
pub static R9: &[u32] = unsafe { from_ptr_range(&D0..(&D0 as *const u32).add(1)) };
pub static R10: &[u32] = unsafe { from_ptr_range(&D0..&D0) };

const D0: u32 = 0x11111111; // Constant chosen for endianness-independent behavior.
const D1: MaybeUninit<&u32> = MaybeUninit::uninit();
const D2: Struct = Struct { a: 1, b: 2, c: 3, d: 4 };
const D3: &u32 = &42;
const D4: [u32; 2] = [17, 42];

#[repr(C)]
struct Struct {
    a: u8,
    // _pad: [MaybeUninit<u8>; 3]
    b: u32,
    c: u16,
    d: u8,
    // _pad: [MaybeUninit<u8>; 1]
}

fn main() {}
