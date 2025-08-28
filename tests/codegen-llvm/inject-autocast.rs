//@ compile-flags: -C opt-level=0
//@ only-x86_64

#![feature(link_llvm_intrinsics, abi_unadjusted, repr_simd, simd_ffi, portable_simd, f16)]
#![crate_type = "lib"]

use std::simd::i64x2;

#[repr(simd)]
pub struct Tile([i8; 1024]);

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

    // CHECK: %2 = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %0, <2 x i64> %1)
    // CHECK-NEXT: %3 = extractvalue { <2 x i1>, <2 x i1> } %2, 0
    // CHECK-NEXT: %4 = shufflevector <2 x i1> %3, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK-NEXT: %5 = bitcast <8 x i1> %4 to i8
    // CHECK-NEXT: %6 = insertvalue { i8, i8 } poison, i8 %5, 0
    // CHECK-NEXT: %7 = extractvalue { <2 x i1>, <2 x i1> } %2, 1
    // CHECK-NEXT: %8 = shufflevector <2 x i1> %7, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
    // CHECK-NEXT: %9 = bitcast <8 x i1> %8 to i8
    // CHECK-NEXT: %10 = insertvalue { i8, i8 } %6, i8 %9, 1
    foo(a, b)
}

// CHECK-LABEL: @struct_autocast
#[no_mangle]
pub unsafe fn struct_autocast(key_metadata: u32, key: i64x2) -> Bar {
    extern "unadjusted" {
        #[link_name = "llvm.x86.encodekey128"]
        fn foo(key_metadata: u32, key: i64x2) -> Bar;
    }

    // CHECK: %1 = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 %key_metadata, <2 x i64> %0)
    // CHECK-NEXT: %2 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 0
    // CHECK-NEXT: %3 = insertvalue %Bar poison, i32 %2, 0
    // CHECK-NEXT: %4 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 1
    // CHECK-NEXT: %5 = insertvalue %Bar %3, <2 x i64> %4, 1
    // CHECK-NEXT: %6 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 2
    // CHECK-NEXT: %7 = insertvalue %Bar %5, <2 x i64> %6, 2
    // CHECK-NEXT: %8 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 3
    // CHECK-NEXT: %9 = insertvalue %Bar %7, <2 x i64> %8, 3
    // CHECK-NEXT: %10 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 4
    // CHECK-NEXT: %11 = insertvalue %Bar %9, <2 x i64> %10, 4
    // CHECK-NEXT: %12 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 5
    // CHECK-NEXT: %13 = insertvalue %Bar %11, <2 x i64> %12, 5
    // CHECK-NEXT: %14 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %1, 6
    // CHECK-NEXT: %15 = insertvalue %Bar %13, <2 x i64> %14, 6
    foo(key_metadata, key)
}

// CHECK-LABEL: @i1_vector_autocast
#[no_mangle]
pub unsafe fn i1_vector_autocast(a: f16x8) -> u8 {
    extern "unadjusted" {
        #[link_name = "llvm.x86.avx512fp16.fpclass.ph.128"]
        fn foo(a: f16x8, b: i32) -> u8;
    }

    // CHECK: %1 = call <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half> %0, i32 1)
    // CHECK-NEXT: %_0 = bitcast <8 x i1> %1 to i8
    foo(a, 1)
}

// CHECK: declare { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64>, <2 x i64>)

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)

// CHECK: declare <8 x i1> @llvm.x86.avx512fp16.fpclass.ph.128(<8 x half>, i32 immarg)
