// needs-sanitizer-support
// needs-sanitizer-address
// ignore-cross-compile
//
// compile-flags: -Z sanitizer=address -O
//
// run-fail
// error-pattern: AddressSanitizer: SEGV

use std::ffi::c_void;

extern "C" {
    fn free(ptr: *mut c_void);
}

fn main() {
    unsafe {
        free(1 as *mut c_void);
    }
}
