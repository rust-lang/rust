//@ compile-flags: -Copt-level=3
//@ only-64bit

#![crate_type = "lib"]

use std::num::NonZero;

// CHECK-LABEL:range(i16 1, 0) i16 @read_unaligned_i16(ptr{{.+}}%ptr)
#[no_mangle]
unsafe fn read_unaligned_i16(ptr: *const NonZero<i16>) -> NonZero<i16> {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load i16, ptr %ptr, align 1
    // CHECK-NOT: !noundef
    // CHECK-NOT: !range
    // CHECK-NEXT: ret i16 [[TEMP]]
    ptr.read_unaligned()
}

// CHECK-LABEL: void @typed_copy_unaligned_i32(ptr{{.+}}%src, ptr{{.+}}%dst)
#[no_mangle]
unsafe fn typed_copy_unaligned_i32(src: *const NonZero<i32>, dst: *mut NonZero<i32>) {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load i32, ptr %src, align 1
    // CHECK-NOT: !noundef
    // CHECK-NOT: !range
    // CHECK-NEXT: store i32 [[TEMP]], ptr %dst, align 1
    // CHECK-NEXT: ret void
    dst.write_unaligned(src.read_unaligned())
}

// CHECK-LABEL: void @write_unaligned_i64(ptr{{.+}}%ptr, i64{{.+}}%val)
#[no_mangle]
unsafe fn write_unaligned_i64(ptr: *mut NonZero<i64>, val: NonZero<i64>) {
    // CHECK: start:
    // CHECK-NEXT: store i64 %val, ptr %ptr, align 1
    // CHECK-NEXT: ret void
    ptr.write_unaligned(val)
}
