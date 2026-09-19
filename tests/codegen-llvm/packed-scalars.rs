//@ revisions: OPT DBG
//@ compile-flags: -C no-prepopulate-passes
//@ [DBG] compile-flags: -C opt-level=0
//@ [OPT] compile-flags: -C opt-level=3

#![crate_type = "lib"]

#[derive(Copy, Clone)]
#[repr(Rust, packed(2))]
pub struct PackedScalar(u64);

// CHECK-LABEL: @read_packed_scalar
#[no_mangle]
pub unsafe fn read_packed_scalar(ptr: *const PackedScalar) -> PackedScalar {
    // CHECK: start
    // CHECK-NEXT: [[TEMP:%.+]] = load i64, ptr %ptr, align 2
    // OPT-SAME: !noundef
    // CHECK-NEXT: ret i64 [[TEMP]]
    *ptr
}

// CHECK-LABEL: @write_packed_scalar
#[no_mangle]
pub unsafe fn write_packed_scalar(ptr: *mut PackedScalar, val: PackedScalar) {
    // CHECK: start
    // CHECK-NEXT: store i64 %val, ptr %ptr, align 2
    // CHECK-NEXT: ret void
    *ptr = val;
}

// CHECK-LABEL: @copy_packed_scalar
#[no_mangle]
pub unsafe fn copy_packed_scalar(dst: *mut PackedScalar, src: *const PackedScalar) {
    // CHECK: start
    // CHECK-NEXT: [[TEMP:%.+]] = load i64, ptr %src, align 2
    // OPT-SAME: !noundef
    // CHECK-NEXT: store i64 [[TEMP]], ptr %dst, align 2
    // CHECK-NEXT: ret void
    *dst = *src;
}
