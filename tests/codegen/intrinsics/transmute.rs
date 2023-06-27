// compile-flags: -O -C no-prepopulate-passes
// only-64bit (so I don't need to worry about usize)
// min-llvm-version: 15.0 # this test assumes `ptr`s

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![feature(inline_const)]
#![allow(unreachable_code)]

use std::intrinsics::{transmute, transmute_unchecked};
use std::mem::MaybeUninit;

// Some of these need custom MIR to not get removed by MIR optimizations.
use std::intrinsics::mir::*;

pub enum ZstNever {}

#[repr(align(2))]
pub struct BigNever(ZstNever, u16, ZstNever);

#[repr(align(8))]
pub struct Scalar64(i64);

#[repr(C, align(4))]
pub struct Aggregate64(u16, u8, i8, f32);

#[repr(C)]
pub struct Aggregate8(u8);

// CHECK-LABEL: @check_bigger_size(
#[no_mangle]
pub unsafe fn check_bigger_size(x: u16) -> u32 {
    // CHECK: call void @llvm.trap
    transmute_unchecked(x)
}

// CHECK-LABEL: @check_smaller_size(
#[no_mangle]
pub unsafe fn check_smaller_size(x: u32) -> u16 {
    // CHECK: call void @llvm.trap
    transmute_unchecked(x)
}

// CHECK-LABEL: @check_smaller_array(
#[no_mangle]
pub unsafe fn check_smaller_array(x: [u32; 7]) -> [u32; 3] {
    // CHECK: call void @llvm.trap
    transmute_unchecked(x)
}

// CHECK-LABEL: @check_bigger_array(
#[no_mangle]
pub unsafe fn check_bigger_array(x: [u32; 3]) -> [u32; 7] {
    // CHECK: call void @llvm.trap
    transmute_unchecked(x)
}

// CHECK-LABEL: @check_to_empty_array(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_to_empty_array(x: [u32; 5]) -> [u32; 0] {
    // CHECK-NOT: trap
    // CHECK: call void @llvm.trap
    // CHECK-NOT: trap
    mir! {
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_from_empty_array(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_from_empty_array(x: [u32; 0]) -> [u32; 5] {
    // CHECK-NOT: trap
    // CHECK: call void @llvm.trap
    // CHECK-NOT: trap
    mir! {
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_to_uninhabited(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_to_uninhabited(x: u16) {
    // CHECK-NOT: trap
    // CHECK: call void @llvm.trap
    // CHECK-NOT: trap
    mir! {
        let temp: BigNever;
        {
            temp = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_from_uninhabited(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_from_uninhabited(x: BigNever) -> u16 {
    // CHECK: ret i16 poison
    mir! {
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_intermediate_passthrough(
#[no_mangle]
pub unsafe fn check_intermediate_passthrough(x: u32) -> i32 {
    // CHECK: start
    // CHECK: %[[TMP:.+]] = add i32 1, %x
    // CHECK: %[[RET:.+]] = add i32 %[[TMP]], 1
    // CHECK: ret i32 %[[RET]]
    unsafe { transmute::<u32, i32>(1 + x) + 1 }
}

// CHECK-LABEL: @check_nop_pair(
#[no_mangle]
pub unsafe fn check_nop_pair(x: (u8, i8)) -> (i8, u8) {
    // CHECK-NOT: alloca
    // CHECK: %0 = insertvalue { i8, i8 } poison, i8 %x.0, 0
    // CHECK: %1 = insertvalue { i8, i8 } %0, i8 %x.1, 1
    // CHECK: ret { i8, i8 } %1
    unsafe { transmute(x) }
}

// CHECK-LABEL: @check_to_newtype(
#[no_mangle]
pub unsafe fn check_to_newtype(x: u64) -> Scalar64 {
    // CHECK-NOT: alloca
    // CHECK: ret i64 %x
    transmute(x)
}

// CHECK-LABEL: @check_from_newtype(
#[no_mangle]
pub unsafe fn check_from_newtype(x: Scalar64) -> u64 {
    // CHECK-NOT: alloca
    // CHECK: ret i64 %x
    transmute(x)
}

// CHECK-LABEL: @check_aggregate_to_bool(
#[no_mangle]
pub unsafe fn check_aggregate_to_bool(x: Aggregate8) -> bool {
    // CHECK: %x = alloca %Aggregate8, align 1
    // CHECK: %[[BYTE:.+]] = load i8, ptr %x, align 1
    // CHECK: %[[BOOL:.+]] = trunc i8 %[[BYTE]] to i1
    // CHECK: ret i1 %[[BOOL]]
    transmute(x)
}

// CHECK-LABEL: @check_aggregate_from_bool(
#[no_mangle]
pub unsafe fn check_aggregate_from_bool(x: bool) -> Aggregate8 {
    // CHECK: %_0 = alloca %Aggregate8, align 1
    // CHECK: %[[BYTE:.+]] = zext i1 %x to i8
    // CHECK: store i8 %[[BYTE]], ptr %_0, align 1
    transmute(x)
}

// CHECK-LABEL: @check_byte_to_bool(
#[no_mangle]
pub unsafe fn check_byte_to_bool(x: u8) -> bool {
    // CHECK-NOT: alloca
    // CHECK: %[[R:.+]] = trunc i8 %x to i1
    // CHECK: ret i1 %[[R]]
    transmute(x)
}

// CHECK-LABEL: @check_byte_from_bool(
#[no_mangle]
pub unsafe fn check_byte_from_bool(x: bool) -> u8 {
    // CHECK-NOT: alloca
    // CHECK: %[[R:.+]] = zext i1 %x to i8
    // CHECK: ret i8 %[[R:.+]]
    transmute(x)
}

// CHECK-LABEL: @check_to_pair(
#[no_mangle]
pub unsafe fn check_to_pair(x: u64) -> Option<i32> {
    // CHECK: %_0 = alloca { i32, i32 }, align 4
    // CHECK: store i64 %x, ptr %_0, align 4
    transmute(x)
}

// CHECK-LABEL: @check_from_pair(
#[no_mangle]
pub unsafe fn check_from_pair(x: Option<i32>) -> u64 {
    // The two arguments are of types that are only 4-aligned, but they're
    // immediates so we can write using the destination alloca's alignment.
    const { assert!(std::mem::align_of::<Option<i32>>() == 4) };

    // CHECK: %_0 = alloca i64, align 8
    // CHECK: store i32 %x.0, ptr %0, align 8
    // CHECK: store i32 %x.1, ptr %1, align 4
    // CHECK: %2 = load i64, ptr %_0, align 8
    // CHECK: ret i64 %2
    transmute(x)
}

// CHECK-LABEL: @check_to_float(
#[no_mangle]
pub unsafe fn check_to_float(x: u32) -> f32 {
    // CHECK-NOT: alloca
    // CHECK: %_0 = bitcast i32 %x to float
    // CHECK: ret float %_0
    transmute(x)
}

// CHECK-LABEL: @check_from_float(
#[no_mangle]
pub unsafe fn check_from_float(x: f32) -> u32 {
    // CHECK-NOT: alloca
    // CHECK: %_0 = bitcast float %x to i32
    // CHECK: ret i32 %_0
    transmute(x)
}

// CHECK-LABEL: @check_to_bytes(
#[no_mangle]
pub unsafe fn check_to_bytes(x: u32) -> [u8; 4] {
    // CHECK: %_0 = alloca [4 x i8], align 1
    // CHECK: store i32 %x, ptr %_0, align 1
    transmute(x)
}

// CHECK-LABEL: @check_from_bytes(
#[no_mangle]
pub unsafe fn check_from_bytes(x: [u8; 4]) -> u32 {
    // CHECK: %x = alloca [4 x i8], align 1
    // CHECK: %[[VAL:.+]] = load i32, ptr %x, align 1
    // CHECK: ret i32 %[[VAL]]
    transmute(x)
}

// CHECK-LABEL: @check_to_aggregate(
#[no_mangle]
pub unsafe fn check_to_aggregate(x: u64) -> Aggregate64 {
    // CHECK: %_0 = alloca %Aggregate64, align 4
    // CHECK: store i64 %x, ptr %_0, align 4
    // CHECK: %0 = load i64, ptr %_0, align 4
    // CHECK: ret i64 %0
    transmute(x)
}

// CHECK-LABEL: @check_from_aggregate(
#[no_mangle]
pub unsafe fn check_from_aggregate(x: Aggregate64) -> u64 {
    // CHECK: %x = alloca %Aggregate64, align 4
    // CHECK: %[[VAL:.+]] = load i64, ptr %x, align 4
    // CHECK: ret i64 %[[VAL]]
    transmute(x)
}

// CHECK-LABEL: @check_long_array_less_aligned(
#[no_mangle]
pub unsafe fn check_long_array_less_aligned(x: [u64; 100]) -> [u16; 400] {
    // CHECK-NEXT: start
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 2 %_0, ptr align 8 %x, i64 800, i1 false)
    // CHECK-NEXT: ret void
    transmute(x)
}

// CHECK-LABEL: @check_long_array_more_aligned(
#[no_mangle]
pub unsafe fn check_long_array_more_aligned(x: [u8; 100]) -> [u32; 25] {
    // CHECK-NEXT: start
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_0, ptr align 1 %x, i64 100, i1 false)
    // CHECK-NEXT: ret void
    transmute(x)
}

// CHECK-LABEL: @check_pair_with_bool(
#[no_mangle]
pub unsafe fn check_pair_with_bool(x: (u8, bool)) -> (bool, i8) {
    // CHECK-NOT: alloca
    // CHECK: trunc i8 %x.0 to i1
    // CHECK: zext i1 %x.1 to i8
    transmute(x)
}

// CHECK-LABEL: @check_float_to_pointer(
#[no_mangle]
pub unsafe fn check_float_to_pointer(x: f64) -> *const () {
    // CHECK-NOT: alloca
    // CHECK: %0 = bitcast double %x to i64
    // CHECK: %_0 = inttoptr i64 %0 to ptr
    // CHECK: ret ptr %_0
    transmute(x)
}

// CHECK-LABEL: @check_float_from_pointer(
#[no_mangle]
pub unsafe fn check_float_from_pointer(x: *const ()) -> f64 {
    // CHECK-NOT: alloca
    // CHECK: %0 = ptrtoint ptr %x to i64
    // CHECK: %_0 = bitcast i64 %0 to double
    // CHECK: ret double %_0
    transmute(x)
}

// CHECK-LABEL: @check_array_to_pair(
#[no_mangle]
pub unsafe fn check_array_to_pair(x: [u8; 16]) -> (i64, u64) {
    // CHECK-NOT: alloca
    // CHECK: %[[FST:.+]] = load i64, ptr %{{.+}}, align 1, !noundef !
    // CHECK: %[[SND:.+]] = load i64, ptr %{{.+}}, align 1, !noundef !
    // CHECK: %[[PAIR0:.+]] = insertvalue { i64, i64 } poison, i64 %[[FST]], 0
    // CHECK: %[[PAIR01:.+]] = insertvalue { i64, i64 } %[[PAIR0]], i64 %[[SND]], 1
    // CHECK: ret { i64, i64 } %[[PAIR01]]
    transmute(x)
}

// CHECK-LABEL: @check_pair_to_array(
#[no_mangle]
pub unsafe fn check_pair_to_array(x: (i64, u64)) -> [u8; 16] {
    // CHECK-NOT: alloca
    // CHECK: store i64 %x.0, ptr %{{.+}}, align 1
    // CHECK: store i64 %x.1, ptr %{{.+}}, align 1
    transmute(x)
}

// CHECK-LABEL: @check_heterogeneous_integer_pair(
#[no_mangle]
pub unsafe fn check_heterogeneous_integer_pair(x: (i32, bool)) -> (bool, u32) {
    // CHECK: store i32 %x.0
    // CHECK: %[[WIDER:.+]] = zext i1 %x.1 to i8
    // CHECK: store i8 %[[WIDER]]

    // CHECK: %[[BYTE:.+]] = load i8
    // CHECK: trunc i8 %[[BYTE:.+]] to i1
    // CHECK: load i32
    transmute(x)
}

// CHECK-LABEL: @check_heterogeneous_float_pair(
#[no_mangle]
pub unsafe fn check_heterogeneous_float_pair(x: (f64, f32)) -> (f32, f64) {
    // CHECK: store double %x.0
    // CHECK: store float %x.1
    // CHECK: %[[A:.+]] = load float
    // CHECK: %[[B:.+]] = load double
    // CHECK: %[[P:.+]] = insertvalue { float, double } poison, float %[[A]], 0
    // CHECK: insertvalue { float, double } %[[P]], double %[[B]], 1
    transmute(x)
}

// CHECK-LABEL: @check_issue_110005(
#[no_mangle]
pub unsafe fn check_issue_110005(x: (usize, bool)) -> Option<Box<[u8]>> {
    // CHECK: store i64 %x.0
    // CHECK: %[[WIDER:.+]] = zext i1 %x.1 to i8
    // CHECK: store i8 %[[WIDER]]
    // CHECK: load ptr
    // CHECK: load i64
    transmute(x)
}

// CHECK-LABEL: @check_pair_to_dst_ref(
#[no_mangle]
pub unsafe fn check_pair_to_dst_ref<'a>(x: (usize, usize)) -> &'a [u8] {
    // CHECK: %_0.0 = inttoptr i64 %x.0 to ptr
    // CHECK: %0 = insertvalue { ptr, i64 } poison, ptr %_0.0, 0
    // CHECK: %1 = insertvalue { ptr, i64 } %0, i64 %x.1, 1
    // CHECK: ret { ptr, i64 } %1
    transmute(x)
}

// CHECK-LABEL: @check_issue_109992(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_issue_109992(x: ()) -> [(); 1] {
    // This uses custom MIR to avoid MIR optimizations having removed ZST ops.

    // CHECK: start
    // CHECK-NEXT: ret void
    mir! {
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_unit_to_never(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_unit_to_never(x: ()) {
    // This uses custom MIR to avoid MIR optimizations having removed ZST ops.

    // CHECK-NOT: trap
    // CHECK: call void @llvm.trap
    // CHECK-NOT: trap
    mir! {
        let temp: ZstNever;
        {
            temp = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_unit_from_never(
#[no_mangle]
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub unsafe fn check_unit_from_never(x: ZstNever) -> () {
    // This uses custom MIR to avoid MIR optimizations having removed ZST ops.

    // CHECK: start
    // CHECK-NEXT: ret void
    mir! {
        {
            RET = CastTransmute(x);
            Return()
        }
    }
}

// CHECK-LABEL: @check_maybe_uninit_pair(i16 %x.0, i64 %x.1)
#[no_mangle]
pub unsafe fn check_maybe_uninit_pair(
    x: (MaybeUninit<u16>, MaybeUninit<u64>),
) -> (MaybeUninit<i64>, MaybeUninit<i16>) {
    // Thanks to `MaybeUninit` this is actually defined behaviour,
    // unlike the examples above with pairs of primitives.

    // CHECK: store i16 %x.0
    // CHECK: store i64 %x.1
    // CHECK: load i64
    // CHECK-NOT: noundef
    // CHECK: load i16
    // CHECK-NOT: noundef
    // CHECK: ret { i64, i16 }
    transmute(x)
}

#[repr(align(8))]
pub struct HighAlignScalar(u8);

// CHECK-LABEL: @check_to_overalign(
#[no_mangle]
pub unsafe fn check_to_overalign(x: u64) -> HighAlignScalar {
    // CHECK: %_0 = alloca %HighAlignScalar, align 8
    // CHECK: store i64 %x, ptr %_0, align 8
    // CHECK: %0 = load i64, ptr %_0, align 8
    // CHECK: ret i64 %0
    transmute(x)
}

// CHECK-LABEL: @check_from_overalign(
#[no_mangle]
pub unsafe fn check_from_overalign(x: HighAlignScalar) -> u64 {
    // CHECK: %x = alloca %HighAlignScalar, align 8
    // CHECK: %[[VAL:.+]] = load i64, ptr %x, align 8
    // CHECK: ret i64 %[[VAL]]
    transmute(x)
}
