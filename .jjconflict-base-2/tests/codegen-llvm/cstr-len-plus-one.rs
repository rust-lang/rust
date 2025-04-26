//@ compile-flags: -Copt-level=3 -Cpanic=abort

#![crate_type = "lib"]
#![feature(cstr_bytes)]

use std::ffi::CStr;

// A `CStr`'s length always fits in an isize after the NUL bit is accounted for

// CHECK-LABEL: @cstr_len_plus_one
#[no_mangle]
pub fn cstr_len_plus_one(s: &CStr) -> bool {
    // CHECK: ret i1 true
    s.bytes().count() + 1 <= isize::MAX as usize
}
