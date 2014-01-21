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
use libc::{c_char, size_t, uintptr_t};

#[cold]
#[lang="fail_"]
pub fn fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    ::rt::begin_unwind_raw(expr, file, line);
}

#[cold]
#[lang="fail_bounds_check"]
pub fn fail_bounds_check(file: *c_char, line: size_t, index: size_t, len: size_t) -> ! {
    let msg = format!("index out of bounds: the len is {} but the index is {}",
                      len as uint, index as uint);
    msg.with_c_str(|buf| fail_(buf, file, line))
}

#[lang="malloc"]
#[inline]
pub unsafe fn local_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    ::rt::local_heap::local_malloc(td, size)
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="free"]
#[inline]
pub unsafe fn local_free(ptr: *c_char) {
    ::rt::local_heap::local_free(ptr);
}
