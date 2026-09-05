// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::MaybeUninit;
use minicore::hint::black_box;
use minicore::mem::transmute_copy;

extern "C" fn foo() -> i32 {
    42
}

// Fn pointer round trip: transmute_copy extern "C" fn() -> i32 to extern "C" fn(), call it,
// then transmute_copy back to the original signature and call again.
// The baseline case: confirms transmute_copy resigns exactly like ordinary transmute.
// CHECK-LABEL-DAG: test_transmute_copy
pub fn test_transmute_copy() {
    let fp: extern "C" fn() -> i32 = foo;
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 42271), ptr %fp, align 8
        // DISC: [[FP_LOAD:%.*]] = load ptr, ptr %fp, align 8
        // DISC: [[FP_INT:%.*]] = ptrtoint ptr [[FP_LOAD]] to i64
        // DISC: [[RESIGN0:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[FP_INT]], i32 0, i64 42271, i32 0, i64 18983)
        // DISC: [[ERASED_PTR:%.*]] = inttoptr i64 [[RESIGN0]] to ptr
        // DISC: store ptr [[ERASED_PTR]], ptr %erased, align 8
        // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}foo, i32 0), ptr %fp, align 8
        let erased: extern "C" fn() = transmute_copy(&fp);
        // DISC: [[CALL0:%.*]] = load ptr, ptr %erased, align 8
        // DISC: call void [[CALL0]]() {{.*}} [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void {{.*}}() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        black_box(erased());
        // DISC: [[ERASED_RELOAD:%.*]] = load ptr, ptr %erased, align 8
        // DISC: [[ERASED_INT:%.*]] = ptrtoint ptr [[ERASED_RELOAD]] to i64
        // DISC: [[RESIGN1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[ERASED_INT]], i32 0, i64 18983, i32 0, i64 42271)
        // DISC: %restored = inttoptr i64 [[RESIGN1]] to ptr
        // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
        let restored: extern "C" fn() -> i32 = transmute_copy(&erased);
        // DISC: %_9 = call i32 %restored() {{.*}} [ "ptrauth"(i32 0, i64 42271) ]
        // NO_DISC: call i32 {{.*}}() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        black_box(restored());
    }
}

// Plain byte-slice -> u32, no fn pointer anywhere. align 1 matching the underlying [4 x u8]
// global's real alignment, not the stronger one of u32.
// CHECK-LABEL-DAG: test_transmute_copy_slice_to_u32
pub fn test_transmute_copy_slice_to_u32() {
    let bytes: &[u8] = &[1, 2, 3, 4];
    unsafe {
        // CHECK: %x = load i32, ptr @{{.*}}, align 1
        // CHECK-NOT: llvm.ptrauth.resign
        let x: u32 = transmute_copy(bytes);
        black_box(x);
    }
}

// Statically known length (4-byte) slice can't supply an 8-byte fn pointer. The static too-short
// check must catch this at compile time and lower it as provable UB, rather than emitting a real
// (out-of-bounds) load.
// CHECK-LABEL-DAG: test_transmute_copy_slice_to_fn_ptr_dst8_src4
pub fn test_transmute_copy_slice_to_fn_ptr_dst8_src4() {
    let bytes: &[u8] = &[1, 2, 3, 4];
    unsafe {
        // CHECK: store i1 true, ptr poison
        let x: extern "C" fn() -> i32 = transmute_copy(bytes);
        black_box(x);
    }
}

// Exact length (8-byte) slice supplies enough bytes for the fn pointer. src is raw bytes, so resign
// from 0-discriminator.
// CHECK-LABEL-DAG: test_transmute_copy_slice_to_fn_ptr_dst8_src8
pub fn test_transmute_copy_slice_to_fn_ptr_dst8_src8() {
    let bytes: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];
    unsafe {
        // DISC: [[SRC:%.*]] = load ptr, ptr @{{.*}}, align 1
        // DISC: [[SRCI:%.*]] = ptrtoint ptr [[SRC]] to i64
        // DISC: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRCI]], i32 0, i64 0, i32 0, i64 42271)
        // DISC: %x = inttoptr i64 [[RESIGN]] to ptr
        // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
        let x: extern "C" fn() -> i32 = transmute_copy(bytes);
        black_box(x);
    }
}

// Source longer than needed (9 bytes for an 8-byte dst). Same 0-discriminator resign.
// CHECK-LABEL-DAG: test_transmute_copy_slice_to_fn_ptr_dst8_src9
pub fn test_transmute_copy_slice_to_fn_ptr_dst8_src9() {
    let bytes: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
    unsafe {
        // DISC: [[SRC:%.*]] = load ptr, ptr @{{.*}}, align 1
        // DISC: [[SRCI:%.*]] = ptrtoint ptr [[SRC]] to i64
        // DISC: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRCI]], i32 0, i64 0, i32 0, i64 42271)
        // DISC: %x = inttoptr i64 [[RESIGN]] to ptr
        let x: extern "C" fn() -> i32 = transmute_copy(bytes);
        black_box(x);
    }
}

// str rather than [u8], same length (4) src and dst.
// CHECK-LABEL-DAG: test_transmute_copy_str_to_u32
pub fn test_transmute_copy_str_to_u32() {
    let s: &str = "abcd";
    unsafe {
        // CHECK: %x = load i32, ptr @{{.*}}, align 1
        // CHECK-NOT: llvm.ptrauth.resign
        let x: u32 = transmute_copy(s);
        black_box(x);
    }
}

// &str is statically known to be too short, transmute_copy must produces the invalid/poison
// value.
// CHECK-LABEL-DAG: test_transmute_copy_str_to_fn_ptr_4byte_str_8_byte_ptr
pub fn test_transmute_copy_str_to_fn_ptr_4byte_str_8_byte_ptr() {
    let s: &str = "abcd";
    unsafe {
        // CHECK: store i1 true, ptr poison
        let x: extern "C" fn() -> i32 = transmute_copy(s);
        black_box(x);
    }
}

// When the &str is 8 bytes wide, transmute_copy works correctly, 0-discriminator resign.
// CHECK-LABEL-DAG: test_transmute_copy_str_to_fn_ptr_8byte_str_8_byte_ptr
pub fn test_transmute_copy_str_to_fn_ptr_8byte_str_8_byte_ptr() {
    let s: &str = "abcd1234";
    unsafe {
        // DISC: [[SRC:%.*]] = load ptr, ptr @{{.*}}, align 1
        // DISC: [[SRCI:%.*]] = ptrtoint ptr [[SRC]] to i64
        // DISC: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRCI]], i32 0, i64 0, i32 0, i64 42271)
        // DISC: %x = inttoptr i64 [[RESIGN]] to ptr
        let x: extern "C" fn() -> i32 = transmute_copy(s);
        black_box(x);
    }
}

// Sized shrinking case: (u64, u64) -> u64. Loads the leading bytes directly.
// CHECK-LABEL-DAG: test_transmute_copy_sized_shrink
pub fn test_transmute_copy_sized_shrink() {
    let pair: (u64, u64) = (1, 2);
    unsafe {
        // CHECK: %x = load i64, ptr %pair, align 8
        let x: u64 = transmute_copy(&pair);
        black_box(x);
    }
}

// No size check, but expect resing.
// CHECK-LABEL-DAG: test_transmute_copy_dynamic_slice_to_fn_ptr
pub fn test_transmute_copy_dynamic_slice_to_fn_ptr(bytes: &[u8]) -> extern "C" fn() -> i32 {
    // Resign still fires correctly even with nothing about the source known statically.
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 42271)
    // CHECK-NOT: unreachable
    unsafe { transmute_copy(bytes) }
}

// Same shape for str, which the static check doesn't cover even in the constant-length case.
// CHECK-LABEL-DAG: test_transmute_copy_dynamic_str_to_fn_ptr
pub fn test_transmute_copy_dynamic_str_to_fn_ptr(s: &str) -> extern "C" fn() -> i32 {
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 42271)
    // CHECK-NOT: unreachable
    unsafe { transmute_copy(s) }
}

// dynamic length no fn pointer type, expect no resigning (and no poison/unreachable as we can't
// reason about the size).
// CHECK-LABEL-DAG: test_transmute_copy_dynamic_slice_to_u32
pub fn test_transmute_copy_dynamic_slice_to_u32(bytes: &[u8]) -> u32 {
    // CHECK-NOT: llvm.ptrauth.resign
    // CHECK-NOT: unreachable
    unsafe { transmute_copy(bytes) }
}

// Regression test for an ICE in codegen_transmute_operand when transmute_copy targets a large
// non-scalar value that cannot be represented as an SSA register.
// Expect a memcpy falling back, rather than attempting a register bit-cast.
// CHECK-LABEL-DAG: test_transmute_copy_large_array
pub fn test_transmute_copy_large_array() {
    let src: [MaybeUninit<u8>; 8192] = [MaybeUninit::uninit(); 8192];
    unsafe {
        // CHECK: call void @llvm.memcpy{{.*}}(ptr align 1 %x, ptr align 1 %src, i64 8192, i1 false)
        // CHECK-NOT: unreachable
        let x: [u8; 8192] = transmute_copy(&src);
        black_box(x);
    }
}

trait Foo {}

// transmute_copy derives the source layout from the static type. For dyn Trait sources, this is a
// placeholder layout with alignment 1. See codegen_transmute_copy_from_place.
// CHECK-LABEL-DAG: test_transmute_copy_dyn_trait_to_u64
pub fn test_transmute_copy_dyn_trait_to_u64(obj: &dyn Foo) -> u64 {
    // This is a know limitation, transmute_copy get's src's from its static layout, which for dyn
    // Trait sources is a placeholder - since align 1. See: codegen_transmute_copy_from_place.
    // CHECK: load i64, ptr %obj.0, align 1
    unsafe { transmute_copy(obj) }
}

#[repr(align(16))]
struct MyType;
trait FooA16 {}
impl FooA16 for MyType {}

// Same here, ideally compiler would use align 16.
// CHECK-LABEL-DAG: test_transmute_copy_dyn_trait_align_16_to_u64
pub fn test_transmute_copy_dyn_trait_align_16_to_u64(obj: &dyn FooA16) -> u64 {
    // CHECK: load i64, ptr %obj.0, align 1
    unsafe { transmute_copy(obj) }
}
