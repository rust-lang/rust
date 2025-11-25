//@ compile-flags: -C opt-level=0
//@ only-x86_64

#![feature(link_llvm_intrinsics, abi_unadjusted, simd_ffi, portable_simd)]
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

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)
