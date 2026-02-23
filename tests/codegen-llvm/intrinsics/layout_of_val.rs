//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z inline-mir
//@ only-64bit (so I don't need to worry about usize)
//@ needs-deterministic-layouts

// Note that the layout algorithm currently puts the align before the size,
// because the *type* for the size doesn't have a niche.  This test may need
// to be updated if the in-memory field order of `Layout` ever changes.

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::alloc::Layout;
use std::intrinsics::layout_of_val;

// CHECK-LABEL: @thin_metadata(
#[no_mangle]
pub unsafe fn thin_metadata(ptr: *const [u32; 2]) -> Layout {
    // CHECK-NOT: alloca
    // CHECK: ret { i64, i64 } { i64 4, i64 8 }
    layout_of_val(ptr)
}

// CHECK-LABEL: @slice_metadata(ptr noundef %ptr.0, i64 noundef %ptr.1)
#[no_mangle]
pub unsafe fn slice_metadata(ptr: *const [u32]) -> Layout {
    // CHECK: [[LAYOUT:%.+]] = alloca [16 x i8], align 8
    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK: [[BYTES:%.+]] = mul nuw nsw i64 %ptr.1, 4
    // CHECK-NEXT: store i64 4, ptr [[LAYOUT]], align 8
    // CHECK-NEXT: [[SIZEP:%.+]] = getelementptr inbounds i8, ptr [[LAYOUT]], i64 8
    // CHECK-NEXT: store i64 [[BYTES]], ptr [[SIZEP]], align 8
    // CHECK-NOT: store
    layout_of_val(ptr)
}

pub struct WithTail<T: ?Sized>([u32; 3], T);

// CHECK-LABEL: @dst_metadata
// CHECK-SAME: (ptr noundef %ptr.0, ptr{{.+}}%ptr.1)
#[no_mangle]
pub unsafe fn dst_metadata(ptr: *const WithTail<dyn std::fmt::Debug>) -> Layout {
    // CHECK: [[LAYOUT:%.+]] = alloca [16 x i8], align 8
    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK: [[DST_SIZEP:%.+]] = getelementptr inbounds i8, ptr %ptr.1, i64 8
    // CHECK-NEXT: [[DST_SIZE:%.+]] = load i64, ptr [[DST_SIZEP]], align 8,
    // CHECK-SAME: !range [[SIZE_RANGE:.+]], !invariant.load
    // CHECK-NEXT: [[DST_ALIGNP:%.+]] = getelementptr inbounds i8, ptr %ptr.1, i64 16
    // CHECK-NEXT: [[DST_ALIGN:%.+]] = load i64, ptr [[DST_ALIGNP]], align 8,
    // CHECK-SAME: !range [[ALIGN_RANGE:!.+]], !invariant.load

    // CHECK-NEXT: [[STRUCT_MORE:%.+]] = icmp ugt i64 4, [[DST_ALIGN]]
    // CHECK-NEXT: [[ALIGN:%.+]] = select i1 [[STRUCT_MORE]], i64 4, i64 [[DST_ALIGN]]

    // CHECK-NEXT: [[MINSIZE:%.+]] = add nuw nsw i64 12, [[DST_SIZE]]
    // CHECK-NEXT: [[ALIGN_M1:%.+]] = sub i64 [[ALIGN]], 1
    // CHECK-NEXT: [[MAXSIZE:%.+]] = add i64 [[MINSIZE]], [[ALIGN_M1]]
    // CHECK-NEXT: [[ALIGN_NEG:%.+]] = sub i64 0, [[ALIGN]]
    // CHECK-NEXT: [[SIZE:%.+]] = and i64 [[MAXSIZE]], [[ALIGN_NEG]]

    // CHECK-NEXT: store i64 [[ALIGN]], ptr [[LAYOUT]], align 8
    // CHECK-NEXT: [[LAYOUT_SIZEP:%.+]] = getelementptr inbounds i8, ptr [[LAYOUT]], i64 8
    // CHECK-NEXT: store i64 [[SIZE]], ptr [[LAYOUT_SIZEP]], align 8

    // CHECK-NOT: store
    // CHECK: load i64, {{.+}} !range [[ALIGNMENT_RANGE:!.+]],
    // CHECK-NOT: store
    layout_of_val(ptr)
}

// CHECK-LABEL: declare

// CHECK: [[ALIGNMENT_RANGE]] = !{i64 1, i64 -[[#0x7FFFFFFFFFFFFFFF]]}
// CHECK: [[SIZE_RANGE]] = !{i64 0, i64 -[[#0x8000000000000000]]}
// CHECK: [[ALIGN_RANGE]] = !{i64 1, i64 [[#0x20000001]]}
