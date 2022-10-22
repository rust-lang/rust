#![warn(clippy::from_raw_with_void_ptr)]

use std::ffi::c_void;

fn main() {
    // must lint
    let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    let _ = unsafe { Box::from_raw(ptr) };

    // shouldn't be linted
    let _ = unsafe { Box::from_raw(ptr as *mut usize) };

    // shouldn't be linted
    let should_not_lint_ptr = Box::into_raw(Box::new(12u8)) as *mut u8;
    let _ = unsafe { Box::from_raw(should_not_lint_ptr as *mut u8) };
}
