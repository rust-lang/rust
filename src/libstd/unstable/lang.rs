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

use cast::transmute;
use libc::{c_char, c_uchar, c_void, size_t, uintptr_t, c_int};
use str;
use sys;
use rt::{context, OldTaskContext};
use rt::task::Task;
use rt::local::Local;
use rt::borrowck;

#[allow(non_camel_case_types)]
pub type rust_task = c_void;

pub mod rustrt {
    use unstable::lang::rust_task;
    use libc::{c_char, uintptr_t};

    extern {
        #[rust_stack]
        pub fn rust_upcall_malloc(td: *c_char, size: uintptr_t) -> *c_char;
        #[rust_stack]
        pub fn rust_upcall_free(ptr: *c_char);
        #[fast_ffi]
        pub fn rust_upcall_malloc_noswitch(td: *c_char, size: uintptr_t)
                                           -> *c_char;
        #[rust_stack]
        pub fn rust_try_get_task() -> *rust_task;
    }
}

#[lang="fail_"]
pub fn fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    sys::begin_unwind_(expr, file, line);
}

#[lang="fail_bounds_check"]
pub fn fail_bounds_check(file: *c_char, line: size_t,
                         index: size_t, len: size_t) {
    let msg = fmt!("index out of bounds: the len is %d but the index is %d",
                    len as int, index as int);
    do msg.as_c_str |buf| {
        fail_(buf, file, line);
    }
}

#[lang="malloc"]
pub unsafe fn local_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    match context() {
        OldTaskContext => {
            return rustrt::rust_upcall_malloc_noswitch(td, size);
        }
        _ => {
            let mut alloc = ::ptr::null();
            do Local::borrow::<Task,()> |task| {
                alloc = task.heap.alloc(td as *c_void, size as uint) as *c_char;
            }
            return alloc;
        }
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

#[lang="strdup_uniq"]
#[inline]
pub unsafe fn strdup_uniq(ptr: *c_uchar, len: uint) -> ~str {
    str::raw::from_buf_len(ptr, len)
}

#[lang="annihilate"]
pub unsafe fn annihilate() {
    ::cleanup::annihilate()
}

#[lang="start"]
pub fn start(main: *u8, argc: int, argv: **c_char,
             crate_map: *u8) -> int {
    use rt;
    use os;

    unsafe {
        let use_old_rt = os::getenv("RUST_OLDRT").is_some();
        if use_old_rt {
            return rust_start(main as *c_void, argc as c_int, argv,
                              crate_map as *c_void) as int;
        } else {
            return do rt::start(argc, argv as **u8, crate_map) {
                let main: extern "Rust" fn() = transmute(main);
                main();
            };
        }
    }

    extern {
        fn rust_start(main: *c_void, argc: c_int, argv: **c_char,
                      crate_map: *c_void) -> c_int;
    }
}
