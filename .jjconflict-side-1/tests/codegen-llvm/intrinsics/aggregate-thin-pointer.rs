//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z mir-enable-passes=-InstSimplify
//@ only-64bit (so I don't need to worry about usize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::aggregate_raw_ptr;

// InstSimplify replaces these with casts if it can, which means they're almost
// never seen in codegen, but PR#121571 found a way, so add a test for it.

#[inline(never)]
pub fn opaque(_p: &*const i32) {}

// CHECK-LABEL: @thin_ptr_via_aggregate(
#[no_mangle]
pub unsafe fn thin_ptr_via_aggregate(p: *const ()) {
    // CHECK: %mem = alloca
    // CHECK: store ptr %p, ptr %mem
    // CHECK: call {{.+}}aggregate_thin_pointer{{.+}} %mem)
    let mem = aggregate_raw_ptr(p, ());
    opaque(&mem);
}
