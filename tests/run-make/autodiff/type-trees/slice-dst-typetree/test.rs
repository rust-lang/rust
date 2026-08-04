#![crate_type = "lib"]
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;
use std::ffi::OsStr;

// Reduced from `clap_lex::OsStrExt::split_once`.
#[no_mangle]
#[inline(never)]
pub fn split_once<'s>(arg: &'s OsStr, needle: &str) -> Option<(&'s OsStr, &'s OsStr)> {
    let bytes = arg.as_encoded_bytes();
    let index = bytes.windows(needle.len()).position(|window| window == needle.as_bytes())?;
    let (first, second) = bytes.split_at(index + needle.len());
    unsafe {
        Some((
            OsStr::from_encoded_bytes_unchecked(first),
            OsStr::from_encoded_bytes_unchecked(second),
        ))
    }
}

#[repr(C)]
pub struct Header<T: ?Sized> {
    tag: f32,
    data: T,
}

#[autodiff_reverse(d_header_sum, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
pub fn header_sum(value: &Header<[f32]>) -> f32 {
    value.tag + value.data.iter().sum::<f32>()
}

#[no_mangle]
pub fn exercise_header_sum(value: &Header<[f32]>, derivative: &mut Header<[f32]>) -> f32 {
    d_header_sum(value, derivative, 1.0)
}

// ZST slice elements yield an empty child TypeTree; element size 0 is expected.
#[autodiff_reverse(d_zst_slice_len, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
pub fn zst_slice_len(slice: &[()]) -> f32 {
    slice.len() as f32
}

#[no_mangle]
pub fn exercise_zst_slice_len(slice: &[()], derivative: &mut [()]) -> f32 {
    d_zst_slice_len(slice, derivative, 1.0)
}
