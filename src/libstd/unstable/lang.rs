// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime calls emitted by the compiler.

use c_str::ToCStr;
use cast::transmute;
use libc::{c_char, c_void, size_t, uintptr_t};
use option::{Option, None, Some};
use sys;
use rt::task::Task;
use rt::local::Local;
use rt::borrowck;

#[lang="fail_"]
pub fn fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    sys::begin_unwind_(expr, file, line);
}

#[lang="fail_bounds_check"]
pub fn fail_bounds_check(file: *c_char, line: size_t,
                         index: size_t, len: size_t) {
    let msg = format!("index out of bounds: the len is {} but the index is {}",
                      len as int, index as int);
    do msg.with_c_str |buf| {
        fail_(buf, file, line);
    }
}

#[lang="malloc"]
pub unsafe fn local_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    // XXX: Unsafe borrow for speed. Lame.
    let task: Option<*mut Task> = Local::try_unsafe_borrow();
    match task {
        Some(task) => {
            (*task).heap.alloc(td as *c_void, size as uint) as *c_char
        }
        None => rtabort!("local malloc outside of task")
    }
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="free"]
pub unsafe fn local_free(ptr: *c_char) {
    ::rt::local_heap::local_free(ptr);
}

#[lang="borrow_as_imm"]
#[inline]
pub unsafe fn borrow_as_imm(a: *u8, file: *c_char, line: size_t) -> uint {
    borrowck::borrow_as_imm(a, file, line)
}

#[lang="borrow_as_mut"]
#[inline]
pub unsafe fn borrow_as_mut(a: *u8, file: *c_char, line: size_t) -> uint {
    borrowck::borrow_as_mut(a, file, line)
}

#[lang="record_borrow"]
pub unsafe fn record_borrow(a: *u8, old_ref_count: uint,
                            file: *c_char, line: size_t) {
    borrowck::record_borrow(a, old_ref_count, file, line)
}

#[lang="unrecord_borrow"]
pub unsafe fn unrecord_borrow(a: *u8, old_ref_count: uint,
                              file: *c_char, line: size_t) {
    borrowck::unrecord_borrow(a, old_ref_count, file, line)
}

#[lang="return_to_mut"]
#[inline]
pub unsafe fn return_to_mut(a: *u8, orig_ref_count: uint,
                            file: *c_char, line: size_t) {
    borrowck::return_to_mut(a, orig_ref_count, file, line)
}

#[lang="check_not_borrowed"]
#[inline]
pub unsafe fn check_not_borrowed(a: *u8,
                                 file: *c_char,
                                 line: size_t) {
    borrowck::check_not_borrowed(a, file, line)
}

#[lang="start"]
pub fn start(main: *u8, argc: int, argv: **c_char) -> int {
    use rt;

    unsafe {
        return do rt::start(argc, argv as **u8) {
            let main: extern "Rust" fn() = transmute(main);
            main();
        };
    }
}
