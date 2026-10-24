//@ compile-flags: -C opt-level=0 -C target-feature=+kl,+avx512vp2intersect,+avx512vl,+avx512dq,+avxneconvert,+amx-int8
//@ only-x86_64

#![feature(link_llvm_intrinsics, simd_ffi, portable_simd, repr_simd)]
#![crate_type = "lib"]

use std::simd::{f32x4, i16x8, i64x2};

pub struct Bar(u32, i64x2, i64x2, i64x2, i64x2, i64x2, i64x2);

#[repr(simd)]
pub struct Tile([i8; 1024]);

// CHECK-LABEL: @struct_autocast
#[no_mangle]
pub unsafe fn struct_autocast(key_metadata: u32, key: i64x2) -> Bar {
    extern "llvm-intrinsic" {
        #[link_name = "llvm.x86.encodekey128"]
        fn foo(key_metadata: u32, key: i64x2) -> Bar;
    }

    // CHECK: %0 = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 %key_metadata, <2 x i64> %key)
    // CHECK-NEXT: %1 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 0
    // CHECK-NEXT: %2 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 1
    // CHECK-NEXT: %3 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 2
    // CHECK-NEXT: %4 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 3
    // CHECK-NEXT: %5 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 4
    // CHECK-NEXT: %6 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 5
    // CHECK-NEXT: %7 = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %0, 6
    // CHECK-NEXT: %8 = getelementptr inbounds i8, ptr %_0, i64 96
    // CHECK-NEXT: store i32 %1, ptr %8, align 16
    // CHECK-NEXT: store <2 x i64> %2, ptr %_0, align 16
    // CHECK-NEXT: %9 = getelementptr inbounds i8, ptr %_0, i64 16
    // CHECK-NEXT: store <2 x i64> %3, ptr %9, align 16
    // CHECK-NEXT: %10 = getelementptr inbounds i8, ptr %_0, i64 32
    // CHECK-NEXT: store <2 x i64> %4, ptr %10, align 16
    // CHECK-NEXT: %11 = getelementptr inbounds i8, ptr %_0, i64 48
    // CHECK-NEXT: store <2 x i64> %5, ptr %11, align 16
    // CHECK-NEXT: %12 = getelementptr inbounds i8, ptr %_0, i64 64
    // CHECK-NEXT: store <2 x i64> %6, ptr %12, align 16
    // CHECK-NEXT: %13 = getelementptr inbounds i8, ptr %_0, i64 80
    // CHECK-NEXT: store <2 x i64> %7, ptr %13, align 16
    foo(key_metadata, key)
}

// CHECK-LABEL: @struct_with_i1_vector_autocast
#[no_mangle]
pub unsafe fn struct_with_i1_vector_autocast(a: i64x2, b: i64x2) -> (u8, u8) {
    extern "llvm-intrinsic" {
        #[link_name = "llvm.x86.avx512.vp2intersect.q.128"]
        fn foo(a: i64x2, b: i64x2) -> (u8, u8);
    }

    // CHECK: %0 = call { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64> %a, <2 x i64> %b)
    // CHECK-NEXT: %1 = extractvalue { <2 x i1>, <2 x i1> } %0, 0
    // CHECK-NEXT: %2 = shufflevector <2 x i1> %1, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
    // CHECK-NEXT: %3 = bitcast <8 x i1> %2 to i8
    // CHECK-NEXT: %4 = extractvalue { <2 x i1>, <2 x i1> } %0, 1
    // CHECK-NEXT: %5 = shufflevector <2 x i1> %4, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
    // CHECK-NEXT: %6 = bitcast <8 x i1> %5 to i8
    // CHECK-NEXT: %7 = insertvalue { i8, i8 } poison, i8 %3, 0
    // CHECK-NEXT: %8 = insertvalue { i8, i8 } %7, i8 %6, 1
    // CHECK-NEXT: %_0.0 = extractvalue { i8, i8 } %8, 0
    // CHECK-NEXT: %_0.1 = extractvalue { i8, i8 } %8, 1
    foo(a, b)
}

// CHECK-LABEL: @i1_vector_autocast
#[no_mangle]
pub unsafe fn i1_vector_autocast(a: u8, b: u8) -> u8 {
    extern "llvm-intrinsic" {
        #[link_name = "llvm.x86.avx512.kadd.b"]
        fn foo(a: u8, b: u8) -> u8;
    }

    // CHECK: [[A:%[0-9]+]] = bitcast i8 {{.*}} to <8 x i1>
    // CHECK: [[B:%[0-9]+]] = bitcast i8 {{.*}} to <8 x i1>
    // CHECK: [[C:%[0-9]+]] = call <8 x i1> @llvm.x86.avx512.kadd.b(<8 x i1> [[A]], <8 x i1> [[B]])
    // CHECK: bitcast <8 x i1> [[C]] to i8
    foo(a, b)
}

// CHECK-LABEL: @bf16_vector_autocast
#[no_mangle]
pub unsafe fn bf16_vector_autocast(a: f32x4) -> i16x8 {
    extern "llvm-intrinsic" {
        #[link_name = "llvm.x86.vcvtneps2bf16128"]
        fn foo(a: f32x4) -> i16x8;
    }

    // CHECK: [[A:%[0-9]+]] = call <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float> {{.*}})
    // CHECK: bitcast <8 x bfloat> [[A]] to <8 x i16>
    foo(a)
}

// CHECK-LABEL: @amx_autocast
#[no_mangle]
pub unsafe fn amx_autocast(m: u16, n: u16, k: u16, a: Tile, b: Tile, c: Tile) -> Tile {
    extern "llvm-intrinsic" {
        #[link_name = "llvm.x86.tdpbuud.internal"]
        fn foo(m: u16, n: u16, k: u16, a: Tile, b: Tile, c: Tile) -> Tile;
    }

    // CHECK: [[A:%[0-9]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> {{.*}})
    // CHECK: [[B:%[0-9]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> {{.*}})
    // CHECK: [[C:%[0-9]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> {{.*}})
    // CHECK: [[D:%[0-9]+]] = call x86_amx @llvm.x86.tdpbuud.internal(i16 %m, i16 %n, i16 %k, x86_amx [[A]], x86_amx [[B]], x86_amx [[C]])
    // CHECK: call <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx [[D]])
    foo(m, n, k, a, b, c)
}

// CHECK: declare { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32, <2 x i64>)

// CHECK: declare { <2 x i1>, <2 x i1> } @llvm.x86.avx512.vp2intersect.q.128(<2 x i64>, <2 x i64>)

// CHECK: declare <8 x i1> @llvm.x86.avx512.kadd.b(<8 x i1>, <8 x i1>)

// CHECK: declare <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float>)

// CHECK: declare x86_amx @llvm.x86.tdpbuud.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)

// CHECK: declare x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8>)

// CHECK: declare <1024 x i8> @llvm.x86.cast.tile.to.vector.v1024i8(x86_amx)
