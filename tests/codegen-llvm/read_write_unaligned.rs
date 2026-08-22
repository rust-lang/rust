//@ compile-flags: -Copt-level=3
//@ only-64bit

#![crate_type = "lib"]

use std::num::NonZero;
use std::ptr::NonNull;

// CHECK-LABEL: nonnull ptr @read_unaligned_ptr(ptr{{.+}}%ptr)
#[no_mangle]
unsafe fn read_unaligned_ptr(ptr: *const NonNull<i16>) -> NonNull<i16> {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load ptr, ptr %ptr, align 1
    // CHECK-SAME: !nonnull
    // CHECK-SAME: !noundef
    // CHECK-NEXT: ret ptr [[TEMP]]
    ptr.read_unaligned()
}

// CHECK-LABEL: range(i16 1, 0) i16 @read_unaligned_i16(ptr{{.+}}%ptr)
#[no_mangle]
unsafe fn read_unaligned_i16(ptr: *const NonZero<i16>) -> NonZero<i16> {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load i16, ptr %ptr, align 1
    // CHECK-SAME: !range [[R16:![0-9]+]]
    // CHECK-SAME: !noundef
    // CHECK-NEXT: ret i16 [[TEMP]]
    ptr.read_unaligned()
}

// CHECK-LABEL: void @typed_copy_unaligned_i32(ptr{{.+}}%src, ptr{{.+}}%dst)
#[no_mangle]
unsafe fn typed_copy_unaligned_i32(src: *const NonZero<i32>, dst: *mut NonZero<i32>) {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load i32, ptr %src, align 1
    // CHECK-SAME: !range [[R32:![0-9]+]]
    // CHECK-SAME: !noundef
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

#[repr(align(128))]
struct HugeBuffer([u64; 1 << 10]);

// CHECK-LABEL: void @read_unaligned_huge(ptr{{.+}}%_0, ptr{{.+}}%ptr)
#[no_mangle]
unsafe fn read_unaligned_huge(ptr: *const HugeBuffer) -> HugeBuffer {
    // CHECK: start:
    // CHECK-NEXT: call void @llvm.memcpy{{.+}} align 128 dereferenceable(8192) %_0, {{.+}} align 1 dereferenceable(8192) %ptr, i64 8192,
    // CHECK-NEXT: ret void
    ptr.read_unaligned()
}

// CHECK-LABEL: void @write_unaligned_huge(ptr{{.+}}%ptr, ptr{{.+}}%val)
#[no_mangle]
unsafe fn write_unaligned_huge(ptr: *mut HugeBuffer, val: HugeBuffer) {
    // CHECK: start:
    // CHECK-NEXT: call void @llvm.memcpy{{.+}} align 1 dereferenceable(8192) %ptr, {{.+}} align 128 dereferenceable(8192) %val, i64 8192,
    // CHECK-NEXT: ret void
    ptr.write_unaligned(val)
}

// CHECK: [[R16]] = !{i16 1, i16 0}
// CHECK: [[R32]] = !{i32 1, i32 0}
