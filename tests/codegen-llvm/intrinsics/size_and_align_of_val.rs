//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z mir-opt-level=0
//@ only-64bit (so I don't need to worry about usize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Here to have unit tests of what they actually emit and to track things like
// <https://github.com/rust-lang/rust/issues/152773>

use std::intrinsics::{align_of_val, size_of_val};

// CHECK-LABEL: @align_of_array(
#[no_mangle]
pub unsafe fn align_of_array(x: &[u16; 7]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 2
    align_of_val(x)
}

// CHECK-LABEL: @size_of_array(
#[no_mangle]
pub unsafe fn size_of_array(x: &[u16; 7]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 14
    size_of_val(x)
}

// CHECK-LABEL: @align_of_slice(
#[no_mangle]
pub unsafe fn align_of_slice(x: &[u16]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 2
    align_of_val(x)
}

// CHECK-LABEL: @size_of_slice(
#[no_mangle]
pub unsafe fn size_of_slice(x: &[u16]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[SIZE:%.+]] = mul nuw nsw i64 %x.1, 2
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

// CHECK-LABEL: @size_of_u8_slice(
#[no_mangle]
pub unsafe fn size_of_u8_slice(x: &[u8]) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 %x.1
    size_of_val(x)
}

#[repr(C)]
pub struct Level0([u8]);
#[repr(C)]
pub struct Level1(u16, Level0);
#[repr(C)]
pub struct Level2(u32, Level1);
#[repr(C)]
pub struct Level3(u64, Level2);

// CHECK-LABEL: @size_of_nested_slice_dst(
#[no_mangle]
pub unsafe fn size_of_nested_slice_dst(x: &Level3) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[UNALIGNED:%.+]] = add nuw i64 %x.1, 21
    // CHECK-NEXT: [[SIZE:%.+]] = and i64 [[UNALIGNED]], -8
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

// CHECK-LABEL: @align_of_nested_slice_dst(
#[no_mangle]
pub unsafe fn align_of_nested_slice_dst(x: &Level3) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 8
    align_of_val(x)
}

#[repr(C)]
pub struct Unpadded0([u64]);
#[repr(C)]
pub struct Unpadded1(u64, Unpadded0);
#[repr(C)]
pub struct Unpadded2(u64, Unpadded1);
#[repr(C)]
pub struct Unpadded3(u64, Unpadded2);

// CHECK-LABEL: @size_of_unpadded_slice_dst(
#[no_mangle]
pub unsafe fn size_of_unpadded_slice_dst(x: &Unpadded3) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[SLICE_BYTES:%.+]] = mul nuw nsw i64 %x.1, 8
    // CHECK-NEXT: [[SIZE:%.+]] = add nuw nsw i64 [[SLICE_BYTES]], 24
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

#[repr(C)]
pub struct AlignedInner(u32, u8, [u8]);
#[repr(C, packed(2))]
pub struct PackedAroundAligned(u8, AlignedInner);
#[repr(C, align(8))]
pub struct AlignedAroundPacked(u8, PackedAroundAligned);

// CHECK-LABEL: @size_of_packed_around_aligned(
#[no_mangle]
pub unsafe fn size_of_packed_around_aligned(x: &PackedAroundAligned) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[UNALIGNED:%.+]] = add nuw i64 %x.1, 8
    // CHECK-NEXT: [[ALIGNED:%.+]] = and i64 [[UNALIGNED]], -4
    // CHECK-NEXT: [[SIZE:%.+]] = add nuw nsw i64 [[ALIGNED]], 2
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

// CHECK-LABEL: @size_of_aligned_around_packed(
#[no_mangle]
pub unsafe fn size_of_aligned_around_packed(x: &AlignedAroundPacked) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[UNALIGNED:%.+]] = add nuw i64 %x.1, 16
    // CHECK-NEXT: [[SIZE:%.+]] = and i64 [[UNALIGNED]], -8
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

#[repr(C)]
pub struct NonPow2ElemSliceDst(u64, [[u8; 3]]);

// CHECK-LABEL: @size_of_non_pow2_elem_slice_dst(
#[no_mangle]
pub unsafe fn size_of_non_pow2_elem_slice_dst(x: &NonPow2ElemSliceDst) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[SLICE_BYTES:%.+]] = mul nuw nsw i64 %x.1, 3
    // CHECK-NEXT: [[UNALIGNED:%.+]] = add nuw i64 [[SLICE_BYTES]], 15
    // CHECK-NEXT: [[SIZE:%.+]] = and i64 [[UNALIGNED]], -8
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

#[repr(C)]
pub struct ZstSliceDst(u32, u8, [()]);

// CHECK-LABEL: @size_of_zst_slice_dst(
#[no_mangle]
pub unsafe fn size_of_zst_slice_dst(x: &ZstSliceDst) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 8
    size_of_val(x)
}

pub trait Trait {}

// CHECK-LABEL: @size_of_manually_drop_dyn(
#[no_mangle]
pub unsafe fn size_of_manually_drop_dyn(x: &std::mem::ManuallyDrop<dyn Trait>) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[VTABLE_SIZE_PTR:%.+]] = getelementptr inbounds i8, ptr %x.1, i64 8
    // CHECK-NEXT: [[SIZE:%.+]] = load i64, ptr [[VTABLE_SIZE_PTR]], align 8, !range [[RANGE:![0-9]+]], !invariant.load
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

#[repr(C)]
pub struct DynWrap0<T: ?Sized>(u8, T);

// CHECK-LABEL: @size_of_dyn_wrap0(
#[no_mangle]
pub unsafe fn size_of_dyn_wrap0(x: &DynWrap0<dyn Trait>) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[VTABLE_SIZE_PTR:%.+]] = getelementptr inbounds i8, ptr %x.1, i64 8
    // CHECK-NEXT: [[DYN_SIZE:%.+]] = load i64, ptr [[VTABLE_SIZE_PTR]], align 8, !range {{![0-9]+}}, !invariant.load
    // CHECK-NEXT: [[VTABLE_ALIGN_PTR:%.+]] = getelementptr inbounds i8, ptr %x.1, i64 16
    // CHECK-NEXT: [[ALIGN:%.+]] = load i64, ptr [[VTABLE_ALIGN_PTR]], align 8, !range {{![0-9]+}}, !invariant.load
    // CHECK-NEXT: [[SIZE:%.+]] = add nuw nsw i64 [[DYN_SIZE]], [[ALIGN]]
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

#[repr(C, packed)]
pub struct PackedDyn<T: ?Sized>(u8, std::mem::ManuallyDrop<T>);

// CHECK-LABEL: @size_of_packed_dyn(
#[no_mangle]
pub unsafe fn size_of_packed_dyn(x: &PackedDyn<dyn Trait>) -> usize {
    // CHECK: start:
    // CHECK-NEXT: [[VTABLE_SIZE_PTR:%.+]] = getelementptr inbounds i8, ptr %x.1, i64 8
    // CHECK-NEXT: [[DYN_SIZE:%.+]] = load i64, ptr [[VTABLE_SIZE_PTR]], align 8, !range {{![0-9]+}}, !invariant.load
    // CHECK-NEXT: [[SIZE:%.+]] = add nuw nsw i64 [[DYN_SIZE]], 1
    // CHECK-NEXT: ret i64 [[SIZE]]
    size_of_val(x)
}

// CHECK-LABEL: @align_of_packed_dyn(
#[no_mangle]
pub unsafe fn align_of_packed_dyn(x: &PackedDyn<dyn Trait>) -> usize {
    // CHECK: start:
    // CHECK-NEXT: ret i64 1
    align_of_val(x)
}
