//@ edition: 2024
#![crate_type = "rlib"]
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::raw_handle;

static PRIVATE: i32 = 7;

#[inline]
pub fn read() -> i32 {
    unsafe { *raw_handle!(PRIVATE).as_ptr() }
}
