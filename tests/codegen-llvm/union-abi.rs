//@ ignore-emscripten vectors passed directly
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
// 32-bit x86 returns `f32` differently to avoid the x87 stack.
// 32-bit systems will return 128bit values using a return area pointer.
//@ revisions: x86-sse x86-nosse bit32 bit64
//@[x86-sse] only-x86
//@[x86-sse] only-rustc_abi-x86-sse2
//@[x86-nosse] only-x86
//@[x86-nosse] ignore-rustc_abi-x86-sse2
//@[bit32] ignore-x86
//@[bit32] only-32bit
//@[bit64] ignore-x86
//@[bit64] only-64bit

// This test that using union forward the abi of the inner type, as
// discussed in #54668

#![crate_type = "lib"]
#![feature(repr_simd)]

#[derive(Copy, Clone)]
pub enum Unhab {}

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i64x4([i64; 4]);

#[derive(Copy, Clone)]
pub union UnionI64x4 {
    a: (),
    b: i64x4,
}

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4(ptr {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4(_: UnionI64x4) {
    loop {}
}

pub union UnionI64x4_ {
    a: i64x4,
    b: (),
    c: i64x4,
    d: Unhab,
    e: ((), ()),
    f: UnionI64x4,
}

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4_(ptr {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4_(_: UnionI64x4_) {
    loop {}
}

pub union UnionI64x4I64 {
    a: i64x4,
    b: i64,
}

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4I64(ptr {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4I64(_: UnionI64x4I64) {
    loop {}
}

pub union UnionI64x4Tuple {
    a: i64x4,
    b: (i64, i64, i64, i64),
}

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4Tuple(ptr {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4Tuple(_: UnionI64x4Tuple) {
    loop {}
}

pub union UnionF32 {
    a: f32,
}

// x86-sse: define {{(dso_local )?}}<4 x i8> @test_UnionF32(float %_1)
// x86-nosse: define {{(dso_local )?}}i32 @test_UnionF32(float %_1)
// bit32: define {{(dso_local )?}}float @test_UnionF32(float %_1)
// bit64: define {{(dso_local )?}}float @test_UnionF32(float %_1)
#[no_mangle]
pub fn test_UnionF32(_: UnionF32) -> UnionF32 {
    loop {}
}

pub union UnionF32F32 {
    a: f32,
    b: f32,
}

// x86-sse: define {{(dso_local )?}}<4 x i8> @test_UnionF32F32(float %_1)
// x86-nosse: define {{(dso_local )?}}i32 @test_UnionF32F32(float %_1)
// bit32: define {{(dso_local )?}}float @test_UnionF32F32(float %_1)
// bit64: define {{(dso_local )?}}float @test_UnionF32F32(float %_1)
#[no_mangle]
pub fn test_UnionF32F32(_: UnionF32F32) -> UnionF32F32 {
    loop {}
}

pub union UnionF32U32 {
    a: f32,
    b: u32,
}

// CHECK: define {{(dso_local )?}}i32 @test_UnionF32U32(i32{{( %0)?}})
#[no_mangle]
pub fn test_UnionF32U32(_: UnionF32U32) -> UnionF32U32 {
    loop {}
}

pub union UnionU128 {
    a: u128,
}
// x86-sse: define {{(dso_local )?}}void @test_UnionU128({{.*}}sret([16 x i8]){{.*}}, i128 %_1)
// x86-nosse: define {{(dso_local )?}}void @test_UnionU128({{.*}}sret([16 x i8]){{.*}}, i128 %_1)
// bit32: define {{(dso_local )?}}void @test_UnionU128({{.*}}sret([16 x i8]){{.*}}, i128 %_1)
// bit64: define {{(dso_local )?}}i128 @test_UnionU128(i128 %_1)
#[no_mangle]
pub fn test_UnionU128(_: UnionU128) -> UnionU128 {
    loop {}
}

#[repr(C)]
pub union CUnionU128 {
    a: u128,
}
// CHECK: define {{(dso_local )?}}void @test_CUnionU128(ptr {{.*}} %_1)
#[no_mangle]
pub fn test_CUnionU128(_: CUnionU128) {
    loop {}
}

pub union UnionBool {
    b: bool,
}
// CHECK: define {{(dso_local )?}}noundef zeroext i1 @test_UnionBool(i8{{.*}} %b)
#[no_mangle]
pub fn test_UnionBool(b: UnionBool) -> bool {
    unsafe { b.b }
}
// CHECK: %_0 = trunc{{( nuw)?}} i8 %b to i1
