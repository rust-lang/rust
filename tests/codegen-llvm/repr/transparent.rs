//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ ignore-riscv64 riscv64 has an i128 type used with test_Vector
//@ ignore-s390x s390x with default march passes vector types per reference
//@ ignore-loongarch64 see codegen/loongarch-abi for loongarch function call tests

// This codegen test embeds assumptions about how certain "C" psABIs are handled
// so it doesn't apply to all architectures or even all OS
// For RISCV: see codegen/riscv-abi
// For LoongArch: see codegen/loongarch-abi

#![crate_type = "lib"]
#![feature(repr_simd, transparent_unions, arm_target_feature, mips_target_feature)]

use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Zst1;
#[derive(Copy, Clone)]
pub struct Zst2(());

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32(f32);

// CHECK: define{{.*}}float @test_F32(float noundef %_1)
#[no_mangle]
pub extern "C" fn test_F32(_: F32) -> F32 {
    loop {}
}

#[repr(transparent)]
pub struct Ptr(*mut u8);

// CHECK: define{{.*}}ptr @test_Ptr(ptr noundef %_1)
#[no_mangle]
pub extern "C" fn test_Ptr(_: Ptr) -> Ptr {
    loop {}
}

#[repr(transparent)]
pub struct WithZst(u64, Zst1);

// CHECK: define{{.*}}i64 @test_WithZst(i64 noundef %_1)
#[no_mangle]
pub extern "C" fn test_WithZst(_: WithZst) -> WithZst {
    loop {}
}

#[repr(transparent)]
pub struct WithZeroSizedArray(*const f32, [i8; 0]);

// CHECK: define{{.*}}ptr @test_WithZeroSizedArray(ptr noundef %_1)
#[no_mangle]
pub extern "C" fn test_WithZeroSizedArray(_: WithZeroSizedArray) -> WithZeroSizedArray {
    loop {}
}

#[repr(transparent)]
pub struct Generic<T>(T);

// CHECK: define{{.*}}double @test_Generic(double noundef %_1)
#[no_mangle]
pub extern "C" fn test_Generic(_: Generic<f64>) -> Generic<f64> {
    loop {}
}

#[repr(transparent)]
pub struct GenericPlusZst<T>(T, Zst2);

#[repr(u8)]
pub enum Bool {
    True,
    False,
    FileNotFound,
}

// CHECK: define{{( dso_local)?}} noundef{{( zeroext)?( range\(i8 0, 3\))?}} i8 @test_Gpz(i8 noundef{{( zeroext)?( range\(i8 0, 3\))?}} %_1)
#[no_mangle]
pub extern "C" fn test_Gpz(_: GenericPlusZst<Bool>) -> GenericPlusZst<Bool> {
    loop {}
}

#[repr(transparent)]
pub struct LifetimePhantom<'a, T: 'a>(*const T, PhantomData<&'a T>);

// CHECK: define{{.*}}ptr @test_LifetimePhantom(ptr noundef %_1)
#[no_mangle]
pub extern "C" fn test_LifetimePhantom(_: LifetimePhantom<i16>) -> LifetimePhantom<i16> {
    loop {}
}

// This works despite current alignment resrictions because PhantomData is always align(1)
#[repr(transparent)]
pub struct UnitPhantom<T, U> {
    val: T,
    unit: PhantomData<U>,
}

pub struct Px;

// CHECK: define{{.*}}float @test_UnitPhantom(float noundef %_1)
#[no_mangle]
pub extern "C" fn test_UnitPhantom(_: UnitPhantom<f32, Px>) -> UnitPhantom<f32, Px> {
    loop {}
}

#[repr(transparent)]
pub struct TwoZsts(Zst1, i8, Zst2);

// CHECK: define{{( dso_local)?}} noundef{{( signext)?}} i8 @test_TwoZsts(i8 noundef{{( signext)?}} %_1)
#[no_mangle]
pub extern "C" fn test_TwoZsts(_: TwoZsts) -> TwoZsts {
    loop {}
}

#[repr(transparent)]
pub struct Nested1(Zst2, Generic<f64>);

// CHECK: define{{.*}}double @test_Nested1(double noundef %_1)
#[no_mangle]
pub extern "C" fn test_Nested1(_: Nested1) -> Nested1 {
    loop {}
}

#[repr(transparent)]
pub struct Nested2(Nested1, Zst1);

// CHECK: define{{.*}}double @test_Nested2(double noundef %_1)
#[no_mangle]
pub extern "C" fn test_Nested2(_: Nested2) -> Nested2 {
    loop {}
}

#[repr(simd)]
struct f32x4([f32; 4]);

#[repr(transparent)]
pub struct Vector(f32x4);

// CHECK: define{{.*}}<4 x float> @test_Vector(<4 x float> %_1)
#[no_mangle]
#[cfg_attr(target_family = "wasm", target_feature(enable = "simd128"))]
#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
#[cfg_attr(target_arch = "mips", target_feature(enable = "msa"))]
pub extern "C" fn test_Vector(_: Vector) -> Vector {
    loop {}
}

trait Mirror {
    type It: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type It = Self;
}

#[repr(transparent)]
pub struct StructWithProjection(<f32 as Mirror>::It);

// CHECK: define{{.*}}float @test_Projection(float noundef %_1)
#[no_mangle]
pub extern "C" fn test_Projection(_: StructWithProjection) -> StructWithProjection {
    loop {}
}

#[repr(transparent)]
pub enum EnumF32 {
    Variant(F32),
}

// CHECK: define{{.*}}float @test_EnumF32(float noundef %_1)
#[no_mangle]
pub extern "C" fn test_EnumF32(_: EnumF32) -> EnumF32 {
    loop {}
}

#[repr(transparent)]
pub enum EnumF32WithZsts {
    Variant(Zst1, F32, Zst2),
}

// CHECK: define{{.*}}float @test_EnumF32WithZsts(float noundef %_1)
#[no_mangle]
pub extern "C" fn test_EnumF32WithZsts(_: EnumF32WithZsts) -> EnumF32WithZsts {
    loop {}
}

#[repr(transparent)]
pub union UnionF32 {
    field: F32,
}

// CHECK: define{{.*}} float @test_UnionF32(float %_1)
#[no_mangle]
pub extern "C" fn test_UnionF32(_: UnionF32) -> UnionF32 {
    loop {}
}

#[repr(transparent)]
pub union UnionF32WithZsts {
    zst1: Zst1,
    field: F32,
    zst2: Zst2,
}

// CHECK: define{{.*}}float @test_UnionF32WithZsts(float %_1)
#[no_mangle]
pub extern "C" fn test_UnionF32WithZsts(_: UnionF32WithZsts) -> UnionF32WithZsts {
    loop {}
}

// All that remains to be tested are aggregates. They are tested in separate files called
// transparent-*.rs  with `only-*` or `ignore-*` directives, because the expected LLVM IR
// function signatures vary so much that it's not reasonably possible to cover all of them with a
// single CHECK line.
//
// You may be wondering why we don't just compare the return types and argument types for equality
// with FileCheck regex captures. Well, rustc doesn't perform newtype unwrapping on newtypes
// containing aggregates. This is OK on all ABIs we support, but because LLVM has not gotten rid of
// pointee types yet, the IR function signature will be syntactically different (%Foo* vs
// %FooWrapper*).
