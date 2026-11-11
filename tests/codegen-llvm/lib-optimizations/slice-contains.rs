// Ensure one-byte bytewise-equality slice `contains` specializations use the optimized byte search.
//@ compile-flags: -Copt-level=3 -Zinline-mir=false

#![crate_type = "lib"]
#![feature(ascii_char)]

use std::ascii::Char as AsciiChar;
use std::cmp::Ordering;
use std::num::{NonZeroI8, NonZeroU8};

// CHECK-LABEL: @contains_bool
#[no_mangle]
pub fn contains_bool(x: bool, data: &[bool]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}

// CHECK-LABEL: @contains_nonzero_u8
#[no_mangle]
pub fn contains_nonzero_u8(x: NonZeroU8, data: &[NonZeroU8]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}

// CHECK-LABEL: @contains_nonzero_i8
#[no_mangle]
pub fn contains_nonzero_i8(x: NonZeroI8, data: &[NonZeroI8], invert: bool) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x) ^ invert
}

// CHECK-LABEL: @contains_option_nonzero_u8
#[no_mangle]
pub fn contains_option_nonzero_u8(x: Option<NonZeroU8>, data: &[Option<NonZeroU8>]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}

// CHECK-LABEL: @contains_option_nonzero_i8
#[no_mangle]
pub fn contains_option_nonzero_i8(
    x: Option<NonZeroI8>,
    data: &[Option<NonZeroI8>],
    invert: bool,
) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x) ^ invert
}

// CHECK-LABEL: @contains_ascii_char
#[no_mangle]
pub fn contains_ascii_char(x: AsciiChar, data: &[AsciiChar]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}

// CHECK-LABEL: @contains_ordering
#[no_mangle]
pub fn contains_ordering(x: Ordering, data: &[Ordering]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}
