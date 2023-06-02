#![warn(clippy::from_raw_with_void_ptr)]
#![allow(clippy::unnecessary_cast)]

use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    // must lint
    let ptr = Box::into_raw(Box::new(42usize)) as *mut c_void;
    let _ = unsafe { Box::from_raw(ptr) };

    // shouldn't be linted
    let _ = unsafe { Box::from_raw(ptr as *mut usize) };

    // shouldn't be linted
    let should_not_lint_ptr = Box::into_raw(Box::new(12u8)) as *mut u8;
    let _ = unsafe { Box::from_raw(should_not_lint_ptr as *mut u8) };

    // must lint
    let ptr = Rc::into_raw(Rc::new(42usize)) as *mut c_void;
    let _ = unsafe { Rc::from_raw(ptr) };

    // must lint
    let ptr = Arc::into_raw(Arc::new(42usize)) as *mut c_void;
    let _ = unsafe { Arc::from_raw(ptr) };

    // must lint
    let ptr = std::rc::Weak::into_raw(Rc::downgrade(&Rc::new(42usize))) as *mut c_void;
    let _ = unsafe { std::rc::Weak::from_raw(ptr) };

    // must lint
    let ptr = std::sync::Weak::into_raw(Arc::downgrade(&Arc::new(42usize))) as *mut c_void;
    let _ = unsafe { std::sync::Weak::from_raw(ptr) };
}
