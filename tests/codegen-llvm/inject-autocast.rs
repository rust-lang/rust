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

// CHECK-LABEL: @struct_autocast
#[no_mangle]
pub unsafe fn struct_autocast(key_metadata: u32, key: i64x2) -> Bar {
    extern "unadjusted" {
        #[link_name = "llvm.x86.encodekey128"]
        fn foo(key_metadata: u32, key: i64x2) -> Bar;
    }

    // CHECK: [[A:%[0-9]+]] = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 {{.*}}, <2 x i64> {{.*}})
    // CHECK: [[B:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 0
    // CHECK: [[C:%[0-9]+]] = insertvalue %Bar poison, i32 [[B]], 0
    // CHECK: [[D:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 1
    // CHECK: [[E:%[0-9]+]] = insertvalue %Bar [[C]], <2 x i64> [[D]], 1
    // CHECK: [[F:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 2
    // CHECK: [[G:%[0-9]+]] = insertvalue %Bar [[E]], <2 x i64> [[F]], 2
    // CHECK: [[H:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 3
    // CHECK: [[I:%[0-9]+]] = insertvalue %Bar [[G]], <2 x i64> [[H]], 3
    // CHECK: [[J:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 4
    // CHECK: [[K:%[0-9]+]] = insertvalue %Bar [[I]], <2 x i64> [[J]], 4
    // CHECK: [[L:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 5
    // CHECK: [[M:%[0-9]+]] = insertvalue %Bar [[K]], <2 x i64> [[L]], 5
    // CHECK: [[N:%[0-9]+]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[A]], 6
    // CHECK:                 insertvalue %Bar [[M]], <2 x i64> [[N]], 6
    foo(key_metadata, key)
}

// CHECK-LABEL: @struct_with_i1_vector_autocast
#[no_mangle]
pub unsafe fn struct_with_i1_vector_autocast(a: i64x2, b: i64x2) -> (u8, u8) {
    extern "unadjusted" {
        #[link_name = "llvm.x86.avx512.vp2intersect.q.128"]
        fn foo(a: i64x2, b: i64x2) -> (u8, u8);
    }

    // CHECK: [[A:%[0-9]+]] = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}})
    // CHECK: [[B:%[0-9]+]] = extractvalue { <2 x i1>, <2 x i1> } [[A]], 0
    // CHECK: [[C:%[0-9]+]] = shufflevector <2 x i1> [[B]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK: [[D:%[0-9]+]] = bitcast <8 x i1> [[C]] to i8
    // CHECK: [[E:%[0-9]+]] = insertvalue { i8, i8 } poison, i8 [[D]], 0
    // CHECK: [[F:%[0-9]+]] = extractvalue { <2 x i1>, <2 x i1> } [[A]], 1
    // CHECK: [[G:%[0-9]+]] = shufflevector <2 x i1> [[F]], <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK: [[H:%[0-9]+]] = bitcast <8 x i1> [[G]] to i8
    // CHECK:                 insertvalue { i8, i8 } [[E]], i8 [[H]], 1
    foo(a, b)
}

// CHECK-LABEL: @i1_vector_autocast
#[no_mangle]
pub unsafe fn i1_vector_autocast(a: f16x8) -> u8 {
    extern "unadjusted" {
        #[link_name = "llvm.x86.avx512fp16.fpclass.ph.128"]
        fn foo(a: f16x8, b: i32) -> u8;
    }

    // CHECK: [[A:%[0-9]+]] = call <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half> {{.*}}, i32 1)
    // CHECK: bitcast <8 x i1> [[A]] to i8
    foo(a, 1)
}

// CHECK-LABEL: @bf16_vector_autocast
#[no_mangle]
pub unsafe fn bf16_vector_autocast(a: f32x4) -> i16x8 {
    extern "unadjusted" {
        #[link_name = "llvm.x86.vcvtneps2bf16128"]
        fn foo(a: f32x4) -> i16x8;
    }

    // CHECK: [[A:%[0-9]+]] = call <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float> {{.*}})
    // CHECK: bitcast <8 x bfloat> [[A]] to <8 x i16>
    foo(a)
}

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)

// CHECK: declare { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64>, <2 x i64>)

// CHECK: declare <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half>, i32 immarg)

// CHECK: declare <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float>)
