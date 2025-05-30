//@ compile-flags: -C opt-level=0
//@ only-x86_64

#![feature(link_llvm_intrinsics, abi_unadjusted, repr_simd, simd_ffi, portable_simd, f16)]
#![crate_type = "lib"]

use std::simd::{f32x4, i16x8, i64x2};

#[repr(C, packed)]
pub struct Bar(u32, i64x2, i64x2, i64x2, i64x2, i64x2, i64x2);
// CHECK: %Bar = type <{ i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }>

#[repr(simd)]
pub struct f16x8([f16; 8]);

// CHECK-LABEL: @struct_with_i1_vector_autocast
#[no_mangle]
pub unsafe fn struct_with_i1_vector_autocast(a: i64x2, b: i64x2) -> (u8, u8) {
    extern "unadjusted" {
        #[link_name = "llvm.x86.avx512.vp2intersect.q.128"]
        fn foo(a: i64x2, b: i64x2) -> (u8, u8);
    }

    // CHECK: %0 = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %a, <2 x i64> %b)
    // CHECK-NEXT: %1 = extractvalue { <2 x i1>, <2 x i1> } %0, 0
    // CHECK-NEXT: %2 = shufflevector <2 x i1> %1, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK-NEXT: %3 = bitcast <8 x i1> %2 to i8
    // CHECK-NEXT: %4 = insertvalue { i8, i8 } poison, i8 %3, 0
    // CHECK-NEXT: %5 = extractvalue { <2 x i1>, <2 x i1> } %0, 1
    // CHECK-NEXT: %6 = shufflevector <2 x i1> %5, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK-NEXT: %7 = bitcast <8 x i1> %6 to i8
    // CHECK-NEXT: %8 = insertvalue { i8, i8 } %4, i8 %7, 1
    foo(a, b)
}

// CHECK-LABEL: @bf16_vector_autocast
#[no_mangle]
pub unsafe fn bf16_vector_autocast(a: f32x4) -> i16x8 {
    extern "unadjusted" {
        #[link_name = "llvm.x86.vcvtneps2bf16128"]
        fn foo(a: f32x4) -> i16x8;
    }

    // CHECK: %0 = call <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float> %a)
    // CHECK-NEXT: %_0 = bitcast <8 x bfloat> %0 to <8 x i16>
    foo(a)
}

// CHECK-LABEL: @struct_autocast
#[no_mangle]
pub unsafe fn struct_autocast(key_metadata: u32, key: i64x2) -> Bar {
    extern "unadjusted" {
        #[link_name = "llvm.x86.encodekey128"]
        fn foo(key_metadata: u32, key: i64x2) -> Bar;
    }

    // CHECK: %0 = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 %key_metadata, <2 x i64> %key)
    // CHECK-NEXT: %1 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 0
    // CHECK-NEXT: %2 = insertvalue %Bar poison, i32 %1, 0
    // CHECK-NEXT: %3 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 1
    // CHECK-NEXT: %4 = insertvalue %Bar %2, <2 x i64> %3, 1
    // CHECK-NEXT: %5 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 2
    // CHECK-NEXT: %6 = insertvalue %Bar %4, <2 x i64> %5, 2
    // CHECK-NEXT: %7 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 3
    // CHECK-NEXT: %8 = insertvalue %Bar %6, <2 x i64> %7, 3
    // CHECK-NEXT: %9 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 4
    // CHECK-NEXT: %10 = insertvalue %Bar %8, <2 x i64> %9, 4
    // CHECK-NEXT: %11 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 5
    // CHECK-NEXT: %12 = insertvalue %Bar %10, <2 x i64> %11, 5
    // CHECK-NEXT: %13 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 6
    // CHECK-NEXT: %14 = insertvalue %Bar %12, <2 x i64> %13, 6
    foo(key_metadata, key)
}

// CHECK-LABEL: @i1_vector_autocast
#[no_mangle]
pub unsafe fn i1_vector_autocast(a: f16x8) -> u8 {
    extern "unadjusted" {
        #[link_name = "llvm.x86.avx512fp16.fpclass.ph.128"]
        fn foo(a: f16x8, b: i32) -> u8;
    }

    // CHECK: %0 = call <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half> %a, i32 1)
    // CHECK-NEXT: %_0 = bitcast <8 x i1> %0 to i8
    foo(a, 1)
}

// CHECK: declare { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64>, <2 x i64>)

// CHECK: declare <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float>)

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)

// CHECK: declare <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half>, i32 immarg)
