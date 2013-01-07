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

use libc::{c_char, c_void, size_t, uintptr_t};
use str;
use sys;

use gc::{cleanup_stack_for_failure, gc, Word};

#[allow(non_camel_case_types)]
pub type rust_task = c_void;

extern mod rustrt {
    #[rust_stack]
    fn rust_upcall_exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char;

    #[rust_stack]
    fn rust_upcall_exchange_free(ptr: *c_char);

    #[rust_stack]
    fn rust_upcall_malloc(td: *c_char, size: uintptr_t) -> *c_char;

    #[rust_stack]
    fn rust_upcall_free(ptr: *c_char);
}

#[rt(fail_)]
#[lang="fail_"]
pub fn rt_fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    sys::begin_unwind_(expr, file, line);
}

#[rt(fail_bounds_check)]
#[lang="fail_bounds_check"]
pub fn rt_fail_bounds_check(file: *c_char, line: size_t,
                        index: size_t, len: size_t) {
    let msg = fmt!("index out of bounds: the len is %d but the index is %d",
                    len as int, index as int);
    do str::as_buf(msg) |p, _len| {
        rt_fail_(p as *c_char, file, line);
    }
}

#[rt(exchange_malloc)]
#[lang="exchange_malloc"]
pub fn rt_exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    return rustrt::rust_upcall_exchange_malloc(td, size);
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[rt(exchange_free)]
#[lang="exchange_free"]
pub fn rt_exchange_free(ptr: *c_char) {
    rustrt::rust_upcall_exchange_free(ptr);
}

#[rt(malloc)]
#[lang="malloc"]
pub fn rt_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    return rustrt::rust_upcall_malloc(td, size);
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[rt(free)]
#[lang="free"]
pub fn rt_free(ptr: *c_char) {
    rustrt::rust_upcall_free(ptr);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
