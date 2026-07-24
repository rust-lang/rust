// Ensure one-byte slice `contains` specializations use the optimized byte search.
//@ compile-flags: -Copt-level=3 -Zinline-mir=false

#![crate_type = "lib"]
#![feature(ascii_char)]

use std::ascii::Char as AsciiChar;
use std::num::NonZeroU8;

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

// CHECK-LABEL: @contains_option_nonzero_u8
#[no_mangle]
pub fn contains_option_nonzero_u8(x: Option<NonZeroU8>, data: &[Option<NonZeroU8>]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}

// CHECK-LABEL: @contains_ascii_char
#[no_mangle]
pub fn contains_ascii_char(x: AsciiChar, data: &[AsciiChar]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}
