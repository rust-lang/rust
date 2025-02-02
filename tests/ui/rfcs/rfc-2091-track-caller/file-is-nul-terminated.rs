//@ run-pass
#![feature(file_with_nul)]

#[track_caller]
const fn assert_file_has_trailing_zero() {
    let caller = core::panic::Location::caller();
    let file_str = caller.file();
    let file_with_nul = caller.file_with_nul();
    if file_str.len() != file_with_nul.count_bytes() {
        panic!("mismatched lengths");
    }
    let trailing_byte: core::ffi::c_char = unsafe {
        *file_with_nul.as_ptr().offset(file_with_nul.count_bytes() as _)
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
