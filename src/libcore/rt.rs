// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[legacy_exports];

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
//! Runtime calls emitted by the compiler.

use cast::transmute;
use libc::{c_char, c_void, size_t, uintptr_t};
use managed::raw::BoxRepr;
use str;
use sys;

use gc::{cleanup_stack_for_failure, gc, Word};

#[allow(non_camel_case_types)]
pub type rust_task = c_void;

#[cfg(target_word_size = "32")]
const FROZEN_BIT: uint = 0x80000000;
#[cfg(target_word_size = "64")]
const FROZEN_BIT: uint = 0x8000000000000000;

extern mod rustrt {
    #[rust_stack]
    unsafe fn rust_upcall_exchange_malloc(td: *c_char, size: uintptr_t)
                                       -> *c_char;

    #[rust_stack]
    unsafe fn rust_upcall_exchange_free(ptr: *c_char);

    #[rust_stack]
    unsafe fn rust_upcall_malloc(td: *c_char, size: uintptr_t) -> *c_char;

    #[rust_stack]
    unsafe fn rust_upcall_free(ptr: *c_char);
}

#[rt(fail_)]
#[lang="fail_"]
pub fn rt_fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    sys::begin_unwind_(expr, file, line);
}

#[rt(fail_bounds_check)]
#[lang="fail_bounds_check"]
pub unsafe fn rt_fail_bounds_check(file: *c_char, line: size_t,
                                   index: size_t, len: size_t) {
    let msg = fmt!("index out of bounds: the len is %d but the index is %d",
                    len as int, index as int);
    do str::as_buf(msg) |p, _len| {
        rt_fail_(p as *c_char, file, line);
    }
}

pub unsafe fn rt_fail_borrowed() {
    let msg = "borrowed";
    do str::as_buf(msg) |msg_p, _| {
        do str::as_buf("???") |file_p, _| {
            rt_fail_(msg_p as *c_char, file_p as *c_char, 0);
        }
    }
}

#[rt(exchange_malloc)]
#[lang="exchange_malloc"]
pub unsafe fn rt_exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    return rustrt::rust_upcall_exchange_malloc(td, size);
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[rt(exchange_free)]
#[lang="exchange_free"]
pub unsafe fn rt_exchange_free(ptr: *c_char) {
    rustrt::rust_upcall_exchange_free(ptr);
}

#[rt(malloc)]
#[lang="malloc"]
pub unsafe fn rt_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    return rustrt::rust_upcall_malloc(td, size);
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[rt(free)]
#[lang="free"]
pub unsafe fn rt_free(ptr: *c_char) {
    rustrt::rust_upcall_free(ptr);
}

#[lang="borrow_as_imm"]
#[inline(always)]
pub unsafe fn borrow_as_imm(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    (*a).header.ref_count |= FROZEN_BIT;
}

#[lang="return_to_mut"]
#[inline(always)]
pub unsafe fn return_to_mut(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    (*a).header.ref_count &= !FROZEN_BIT;
}

#[lang="check_not_borrowed"]
#[inline(always)]
pub unsafe fn check_not_borrowed(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    if ((*a).header.ref_count & FROZEN_BIT) != 0 {
        rt_fail_borrowed();
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
