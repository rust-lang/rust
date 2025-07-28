//@ add-core-stubs
//@ compile-flags: -C opt-level=0 -C no-prepopulate-passes --target=x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, repr_simd)]
#![no_core]
extern crate minicore;

use minicore::*;

// With opaque ptrs in LLVM, `transmute` can load/store any `alloca` as any type,
// without needing to pointercast, and SRoA will turn that into a `bitcast`.
// Thus memory-to-memory transmutes don't need to generate them ourselves.

// However, `bitcast`s and `ptrtoint`s and `inttoptr`s are still worth doing when
// that allows us to avoid the `alloca`s entirely; see `rvalue_creates_operand`.

// CHECK-LABEL: define{{.*}}i32 @f32_to_bits(float %x)
// CHECK: %_0 = bitcast float %x to i32
// CHECK-NEXT: ret i32 %_0
#[no_mangle]
pub fn f32_to_bits(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

// CHECK-LABEL: define{{.*}}i8 @bool_to_byte(i1 zeroext %b)
// CHECK: %_0 = zext i1 %b to i8
// CHECK-NEXT: ret i8 %_0
#[no_mangle]
pub fn bool_to_byte(b: bool) -> u8 {
    unsafe { mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}zeroext i1 @byte_to_bool(i8{{.*}} %byte)
// CHECK: %_0 = trunc{{( nuw)?}} i8 %byte to i1
// CHECK-NEXT: ret i1 %_0
#[no_mangle]
pub unsafe fn byte_to_bool(byte: u8) -> bool {
    mem::transmute(byte)
}

// CHECK-LABEL: define{{.*}}ptr @ptr_to_ptr(ptr %p)
// CHECK: ret ptr %p
#[no_mangle]
pub fn ptr_to_ptr(p: *mut u16) -> *mut u8 {
    unsafe { mem::transmute(p) }
}

// CHECK: define{{.*}}[[USIZE:i[0-9]+]] @ptr_to_int(ptr %p)
// CHECK: %_0 = ptrtoint ptr %p to [[USIZE]]
// CHECK-NEXT: ret [[USIZE]] %_0
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { mem::transmute(p) }
}

// CHECK: define{{.*}}ptr @int_to_ptr([[USIZE]] %i)
// CHECK: %_0 = getelementptr i8, ptr null, [[USIZE]] %i
// CHECK-NEXT: ret ptr %_0
#[no_mangle]
pub fn int_to_ptr(i: usize) -> *mut u16 {
    unsafe { mem::transmute(i) }
}

// This is the one case where signedness matters to transmuting:
// the LLVM type is `i8` here because of `repr(i8)`,
// whereas below with the `repr(u8)` it's `i1` in LLVM instead.
#[repr(i8)]
pub enum FakeBoolSigned {
    False = 0,
    True = 1,
}

// CHECK-LABEL: define{{.*}}i8 @bool_to_fake_bool_signed(i1 zeroext %b)
// CHECK: %_0 = zext i1 %b to i8
// CHECK-NEXT: ret i8 %_0
#[no_mangle]
pub fn bool_to_fake_bool_signed(b: bool) -> FakeBoolSigned {
    unsafe { mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}i1 @fake_bool_signed_to_bool(i8 %b)
// CHECK: %_0 = trunc nuw i8 %b to i1
// CHECK-NEXT: ret i1 %_0
#[no_mangle]
pub fn fake_bool_signed_to_bool(b: FakeBoolSigned) -> bool {
    unsafe { mem::transmute(b) }
}

#[repr(u8)]
pub enum FakeBoolUnsigned {
    False = 0,
    True = 1,
}

// CHECK-LABEL: define{{.*}}i1 @bool_to_fake_bool_unsigned(i1 zeroext %b)
// CHECK: ret i1 %b
#[no_mangle]
pub fn bool_to_fake_bool_unsigned(b: bool) -> FakeBoolUnsigned {
    unsafe { mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}i1 @fake_bool_unsigned_to_bool(i1 zeroext %b)
// CHECK: ret i1 %b
#[no_mangle]
pub fn fake_bool_unsigned_to_bool(b: FakeBoolUnsigned) -> bool {
    unsafe { mem::transmute(b) }
}

#[repr(simd)]
struct S([i64; 1]);

// CHECK-LABEL: define{{.*}}i64 @single_element_simd_to_scalar(<1 x i64> %b)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[RET:.+]] = alloca [8 x i8]
// CHECK-NEXT: store <1 x i64> %b, ptr %[[RET]]
// CHECK-NEXT: %[[TEMP:.+]] = load i64, ptr %[[RET]]
// CHECK-NEXT: ret i64 %[[TEMP]]
#[no_mangle]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
pub extern "C" fn single_element_simd_to_scalar(b: S) -> i64 {
    unsafe { mem::transmute(b) }
}

// CHECK-LABEL: define{{.*}}<1 x i64> @scalar_to_single_element_simd(i64 %b)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[RET:.+]] = alloca [8 x i8]
// CHECK-NEXT: store i64 %b, ptr %[[RET]]
// CHECK-NEXT: %[[TEMP:.+]] = load <1 x i64>, ptr %[[RET]]
// CHECK-NEXT: ret <1 x i64> %[[TEMP]]
#[no_mangle]
#[cfg_attr(target_arch = "x86", target_feature(enable = "sse"))]
pub extern "C" fn scalar_to_single_element_simd(b: i64) -> S {
    unsafe { mem::transmute(b) }
}
