//@ compile-flags: -C opt-level=0
//@ only-x86_64

#![feature(link_llvm_intrinsics, abi_unadjusted, repr_simd, simd_ffi, portable_simd, f16)]
#![crate_type = "lib"]

use std::simd::i64x2;
#[repr(C, packed)]
pub struct Bar(u32, i64x2, i64x2, i64x2, i64x2, i64x2, i64x2);
// CHECK: %Bar = type <{ i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> }>

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

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)
