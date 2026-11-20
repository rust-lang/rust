//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Ensure that we do not call `memcpy` for the following function.
// `memset` and `init` should be called directly on the return pointer.
#[no_mangle]
pub fn nrvo(init: fn(&mut [u8; 4096])) -> [u8; 4096] {
    // CHECK-LABEL: nrvo
    // CHECK: @llvm.memset
    // FIXME: turn on nrvo then check-not: @llvm.memcpy
    // CHECK: ret
    // CHECK-EMPTY
    let mut buf = [0; 4096];
    init(&mut buf);
    buf
}
