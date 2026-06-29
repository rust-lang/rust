//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//
//@ compile-flags: -C sanitizer=address -O -C unsafe-allow-abi-mismatch=sanitizer -Z unstable-options
//
//@ run-fail-or-crash
//@ regex-error-pattern: AddressSanitizer: (SEGV|attempting free on address which was not malloc)

use std::ffi::c_void;

extern "C" {
    fn free(ptr: *mut c_void);
}

fn main() {
    unsafe {
        free(1 as *mut c_void);
    }
}
