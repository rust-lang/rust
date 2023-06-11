// ignore-emscripten vectors passed directly
// compile-flags: -O -C no-prepopulate-passes

// This test that using union forward the abi of the inner type, as
// discussed in #54668

#![crate_type="lib"]
#![feature(repr_simd)]

#[derive(Copy, Clone)]
pub enum Unhab {}

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct i64x4(i64, i64, i64, i64);

#[derive(Copy, Clone)]
pub union UnionI64x4{ a:(), b: i64x4 }

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4({{<4 x i64>\*|ptr}} {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4(_: UnionI64x4) { loop {} }

pub union UnionI64x4_{ a: i64x4, b: (), c:i64x4, d: Unhab, e: ((),()), f: UnionI64x4 }

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4_({{<4 x i64>\*|ptr}} {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4_(_: UnionI64x4_) { loop {} }

pub union UnionI64x4I64{ a: i64x4, b: i64 }

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4I64({{%UnionI64x4I64\*|ptr}} {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4I64(_: UnionI64x4I64) { loop {} }

pub union UnionI64x4Tuple{ a: i64x4, b: (i64, i64, i64, i64) }

// CHECK: define {{(dso_local )?}}void @test_UnionI64x4Tuple({{%UnionI64x4Tuple\*|ptr}} {{.*}} %_1)
#[no_mangle]
pub fn test_UnionI64x4Tuple(_: UnionI64x4Tuple) { loop {} }


pub union UnionF32{a:f32}

// CHECK: define {{(dso_local )?}}float @test_UnionF32(float %_1)
#[no_mangle]
pub fn test_UnionF32(_: UnionF32) -> UnionF32 { loop {} }

pub union UnionF32F32{a:f32, b:f32}

// CHECK: define {{(dso_local )?}}float @test_UnionF32F32(float %_1)
#[no_mangle]
pub fn test_UnionF32F32(_: UnionF32F32) -> UnionF32F32 { loop {} }

pub union UnionF32U32{a:f32, b:u32}

// CHECK: define {{(dso_local )?}}i32 @test_UnionF32U32(i32{{( %0)?}})
#[no_mangle]
pub fn test_UnionF32U32(_: UnionF32U32) -> UnionF32U32 { loop {} }

pub union UnionU128{a:u128}
// CHECK: define {{(dso_local )?}}i128 @test_UnionU128(i128 %_1)
#[no_mangle]
pub fn test_UnionU128(_: UnionU128) -> UnionU128 { loop {} }

#[repr(C)]
pub union CUnionU128{a:u128}
// CHECK: define {{(dso_local )?}}void @test_CUnionU128({{%CUnionU128\*|ptr}} {{.*}} %_1)
#[no_mangle]
pub fn test_CUnionU128(_: CUnionU128) { loop {} }

pub union UnionBool { b:bool }
// CHECK: define {{(dso_local )?}}noundef zeroext i1 @test_UnionBool(i8 %b)
#[no_mangle]
pub fn test_UnionBool(b: UnionBool) -> bool { unsafe { b.b }  }
// CHECK: %0 = trunc i8 %b to i1
