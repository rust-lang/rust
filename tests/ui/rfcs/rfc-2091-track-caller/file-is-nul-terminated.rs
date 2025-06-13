//@ run-pass
#![feature(file_with_nul)]

use std::ffi::CStr;

#[track_caller]
const fn assert_file_has_trailing_zero() {
    let caller = core::panic::Location::caller();
    let file_str = caller.file();
    let file_ptr = caller.file_ptr();
    let file_with_nul = unsafe { CStr::from_ptr(file_ptr) };
    if file_str.len() != file_with_nul.count_bytes() {
        panic!("mismatched lengths");
    }
    let trailing_byte = unsafe {
        *file_ptr.add(file_with_nul.count_bytes())
    };
    if trailing_byte != 0 {
        panic!("trailing byte was nonzero")
    }
}

#[allow(dead_code)]
const _: () = assert_file_has_trailing_zero();

fn main() {
    assert_file_has_trailing_zero();
}
