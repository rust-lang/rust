// run-pass
use std::ptr;

#[repr(C)]
struct Struct {
    a: u32,
    b: u32,
    c: u32,
}
static S: Struct = Struct { a: 0, b: 0, c: 0 };

// For these tests we use offset_from to check that two pointers are equal.
// Rust doesn't currently support comparing pointers in const fn.

static OFFSET_NO_CHANGE: bool = unsafe {
    let p1 = &S.b as *const u32;
    let p2 = p1.offset(2).offset(-2);
    p1.offset_from(p2) == 0
};
static OFFSET_MIDDLE: bool = unsafe {
    let p1 = (&S.a as *const u32).offset(1);
    let p2 = (&S.c as *const u32).offset(-1);
    p1.offset_from(p2) == 0
};
// Pointing to the end of the allocation is OK
static OFFSET_END: bool = unsafe {
    let p1 = (&S.a as *const u32).offset(3);
    let p2 = (&S.c as *const u32).offset(1);
    p1.offset_from(p2) == 0
};
// Casting though a differently sized type is OK
static OFFSET_U8_PTR: bool = unsafe {
    let p1 = (&S.a as *const u32 as *const u8).offset(5);
    let p2 = (&S.c as *const u32 as *const u8).offset(-3);
    p1.offset_from(p2) == 0
};
// Any offset with a ZST does nothing
const OFFSET_ZST: bool = unsafe {
    let pz = &() as *const ();
    // offset_from can't work with ZSTs, so cast to u8 ptr
    let p1 = pz.offset(5) as *const u8;
    let p2 = pz.offset(isize::MIN) as *const u8;
    p1.offset_from(p2) == 0
};
const OFFSET_ZERO: bool = unsafe {
    let p = [0u8; 0].as_ptr();
    p.offset(0).offset_from(p) == 0
};
const OFFSET_ONE: bool = unsafe {
    let p = &42u32 as *const u32;
    p.offset(1).offset_from(p) == 1
};
const OFFSET_DANGLING: bool = unsafe {
    let p = ptr::NonNull::<u8>::dangling().as_ptr();
    p.offset(0).offset_from(p) == 0
};
const OFFSET_UNALIGNED: bool = unsafe {
    let arr = [0u8; 32];
    let p1 = arr.as_ptr();
    let p2 = (p1.offset(2) as *const u32).offset(1);
    (p2 as *const u8).offset_from(p1) == 6
};

const WRAP_OFFSET_NO_CHANGE: bool = unsafe {
    let p1 = &42u32 as *const u32;
    let p2 = p1.wrapping_offset(1000).wrapping_offset(-1000);
    let p3 = p1.wrapping_offset(-1000).wrapping_offset(1000);
    (p1.offset_from(p2) == 0) & (p1.offset_from(p3) == 0)
};
const WRAP_ADDRESS_SPACE: bool = unsafe {
    let p1 = &42u8 as *const u8;
    let p2 = p1.wrapping_offset(isize::MIN).wrapping_offset(isize::MIN);
    p1.offset_from(p2) == 0
};
// Wrap on the count*size_of::<T>() calculation.
const WRAP_SIZE_OF: bool = unsafe {
    // Make sure that if p1 moves backwards, we are still in range
    let arr = [0u32; 2];
    let p = &arr[1] as *const u32;
    // With wrapping arithmetic, isize::MAX * 4 == -4
    let wrapped = p.wrapping_offset(isize::MAX);
    let backward = p.wrapping_offset(-1);
    wrapped.offset_from(backward) == 0
};
const WRAP_INTEGER_POINTER: bool = unsafe {
    let p1 = (0x42 as *const u32).wrapping_offset(4);
    let p2 = 0x52 as *const u32;
    p1.offset_from(p2) == 0
};
const WRAP_NULL: bool = unsafe {
    let p1 = ptr::null::<u32>().wrapping_offset(1);
    let p2 = 0x4 as *const u32;
    p1.offset_from(p2) == 0
};

fn main() {
    assert!(OFFSET_NO_CHANGE);
    assert!(OFFSET_MIDDLE);
    assert!(OFFSET_END);
    assert!(OFFSET_U8_PTR);
    assert!(OFFSET_ZST);
    assert!(OFFSET_ZERO);
    assert!(OFFSET_ONE);
    assert!(OFFSET_DANGLING);
    assert!(OFFSET_UNALIGNED);

    assert!(WRAP_OFFSET_NO_CHANGE);
    assert!(WRAP_ADDRESS_SPACE);
    assert!(WRAP_SIZE_OF);
    assert!(WRAP_INTEGER_POINTER);
    assert!(WRAP_NULL);
}
