//@ run-pass

#![allow(dead_code)]

use std::mem::{size_of, align_of};
use std::os::raw::c_int;

// The two enums that follow are designed so that bugs trigger layout optimization.
// Specifically, if either of the following reprs used here is not detected by the compiler,
// then the sizes will be wrong.

#[repr(C, u8)]
enum E1 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(u8, C)]
enum E2 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

// Check that repr(int) and repr(C) are in fact different from the above

#[repr(u8)]
enum E3 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(u16)]
enum E4 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(u32)]
enum E5 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(u64)]
enum E6 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

#[repr(C)]
enum E7 {
    A(u8, u16, u8),
    B(u8, u16, u8)
}

// From pr 37429

#[repr(C,packed)]
pub struct p0f_api_query {
    pub magic: u32,
    pub addr_type: u8,
    pub addr: [u8; 16],
}

pub fn main() {
    assert_eq!(size_of::<E1>(), 8);
    assert_eq!(size_of::<E2>(), 8);
    assert_eq!(size_of::<E3>(), 6);
    assert_eq!(size_of::<E4>(), 8);
    assert_eq!(size_of::<E5>(), align_size(10, align_of::<u32>()));
    assert_eq!(size_of::<E6>(), align_size(14, align_of::<u64>()));
    assert_eq!(size_of::<E7>(), align_size(6 + c_enum_min_size(), align_of::<c_int>()));
    assert_eq!(size_of::<p0f_api_query>(), 21);
}

fn align_size(size: usize, align: usize) -> usize {
    if size % align != 0 {
        size + (align - (size % align))
    } else {
        size
    }
}

// this is `TargetOptions.c_enum_min_bits` which is not available as a `cfg` value so we retrieve
// the value at runtime. On most targets this is `sizeof(c_int)` but on `thumb*-none` is 1 byte
fn c_enum_min_size() -> usize {
    #[repr(C)]
    enum E {
        A,
    }
    size_of::<E>()
}
