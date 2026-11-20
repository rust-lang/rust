// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// The explicit pointer authentication intrinsics:
// * `ptrauth_c_cast` resigns a code address between the C encodings of two types: a function
//   pointer type uses its type discriminator, a raw pointer or an integer uses 0. Without type
//   discrimination every encoding is 0, so nothing is resigned.
// * `ptrauth_fn_ptr_discriminator` is the type discriminator of a function pointer type.
// * `ptrauth_strip` removes the signature of a code address.
//
// Discriminators:
// * `extern "C" fn()` = 18983
// * `extern "C" fn(i32)` = 2712.

#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::Option;

#[rustc_intrinsic]
unsafe fn ptrauth_c_cast<Src, Dst>(src: Src) -> Dst;

#[rustc_intrinsic]
#[rustc_comptime]
fn ptrauth_fn_ptr_discriminator<F>() -> u64;

#[rustc_intrinsic]
fn ptrauth_strip<T>(ptr: T) -> usize;

type F0 = extern "C" fn();
type F1 = extern "C" fn(i32);

// A `void *` from C (e.g. `dlsym`) carries discriminator 0. It may be null, so the resign is
// guarded by a null check.
// CHECK-LABEL: @from_void_ptr(
#[no_mangle]
pub unsafe fn from_void_ptr(p: *const ()) -> Option<F1> {
    // DISC: icmp eq ptr %p, null
    // DISC: br i1 %{{.*}}, label %ptrauth.null, label %ptrauth.resign
    // DISC: ptrauth.resign:
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
    // DISC: ptrauth.end:
    // NO_DISC-NOT: @llvm.ptrauth.resign
    // NO_DISC-NOT: ptrauth.null
    // CHECK: ret ptr
    unsafe { ptrauth_c_cast::<*const (), Option<F1>>(p) }
}

// A non-null function pointer handed to C as a `void *`: no null check.
// CHECK-LABEL: @to_void_ptr(
#[no_mangle]
pub unsafe fn to_void_ptr(f: F1) -> *const () {
    // DISC-NOT: ptrauth.null
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 0)
    // NO_DISC-NOT: @llvm.ptrauth.resign
    // CHECK: ret ptr
    unsafe { ptrauth_c_cast::<F1, *const ()>(f) }
}

// A generic callback slot (GLib's `GCallback`, Vulkan's `PFN_vkVoidFunction`): resign between
// the discriminators of the real and the generic function pointer type.
// CHECK-LABEL: @to_generic_callback(
#[no_mangle]
pub unsafe fn to_generic_callback(f: F1) -> F0 {
    // DISC-NOT: ptrauth.null
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
    // NO_DISC-NOT: @llvm.ptrauth.resign
    // CHECK: ret ptr
    unsafe { ptrauth_c_cast::<F1, F0>(f) }
}

// CHECK-LABEL: @from_generic_callback(
#[no_mangle]
pub unsafe fn from_generic_callback(g: Option<F0>) -> Option<F1> {
    // DISC: br i1 %{{.*}}, label %ptrauth.null, label %ptrauth.resign
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 18983, i32 0, i64 2712)
    // NO_DISC-NOT: @llvm.ptrauth.resign
    // CHECK: ret ptr
    unsafe { ptrauth_c_cast::<Option<F0>, Option<F1>>(g) }
}

// An integer from C (`uintptr_t`) carries discriminator 0.
// CHECK-LABEL: @from_integer(
#[no_mangle]
pub unsafe fn from_integer(addr: usize) -> Option<F0> {
    // DISC: inttoptr i64 %addr to ptr
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 18983)
    // NO_DISC-NOT: @llvm.ptrauth.resign
    // CHECK: ret ptr
    unsafe { ptrauth_c_cast::<usize, Option<F0>>(addr) }
}

// The same encoding on both sides: nothing to resign.
// CHECK-LABEL: @same_encoding(
#[no_mangle]
pub unsafe fn same_encoding(p: *const ()) -> usize {
    // CHECK-NOT: @llvm.ptrauth.resign
    // CHECK: ptrtoint ptr %p to i64
    // CHECK-NOT: @llvm.ptrauth.resign
    // CHECK: ret i64
    unsafe { ptrauth_c_cast::<*const (), usize>(p) }
}

// CHECK-LABEL: @same_fn_type(
#[no_mangle]
pub unsafe fn same_fn_type(f: F1) -> Option<F1> {
    // CHECK-NOT: @llvm.ptrauth.resign
    // CHECK: ret ptr %f
    unsafe { ptrauth_c_cast::<F1, Option<F1>>(f) }
}

// The resigned pointer is called through its type: the call authenticates with that type.
// CHECK-LABEL: @call_from_void_ptr(
#[no_mangle]
pub unsafe fn call_from_void_ptr(p: *const ()) {
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
    // DISC: call void %{{.*}}(i32 7) {{.*}}[ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void %{{.*}}(i32 7) {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    let f = unsafe { ptrauth_c_cast::<*const (), F1>(p) };
    f(7);
}

// CHECK-LABEL: @strip(
#[no_mangle]
pub fn strip(f: F1) -> usize {
    // CHECK: [[ADDR:%.*]] = ptrtoint ptr %f to i64
    // CHECK: [[STRIPPED:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[ADDR]], i32 0)
    // CHECK: ret i64 [[STRIPPED]]
    ptrauth_strip(f)
}

// The discriminator does not depend on whether type discrimination is enabled, like Clang's
// `__builtin_ptrauth_type_discriminator`.
// CHECK-LABEL: @discriminator_f1(
#[no_mangle]
pub fn discriminator_f1() -> u64 {
    // CHECK: ret i64 2712
    const { ptrauth_fn_ptr_discriminator::<F1>() }
}

// CHECK-LABEL: @discriminator_option_f0(
#[no_mangle]
pub fn discriminator_option_f0() -> u64 {
    // CHECK: ret i64 18983
    const { ptrauth_fn_ptr_discriminator::<Option<F0>>() }
}

// Not an `extern "C"` function pointer: 0.
// CHECK-LABEL: @discriminator_rust_abi(
#[no_mangle]
pub fn discriminator_rust_abi() -> u64 {
    // CHECK: ret i64 0
    const { ptrauth_fn_ptr_discriminator::<fn(i32)>() }
}

// CHECK-LABEL: @discriminator_nested_option(
#[no_mangle]
pub fn discriminator_nested_option() -> u64 {
    // CHECK: ret i64 0
    const { ptrauth_fn_ptr_discriminator::<Option<Option<F0>>>() }
}

extern "C" fn item() {}

const fn discriminator_of_val<F>(_: &F) -> u64 {
    const { ptrauth_fn_ptr_discriminator::<F>() }
}

// A function item type is zero-sized, so it does not hold a code address either; it needs to be
// coerced to a function pointer first.
// CHECK-LABEL: @discriminator_fn_item(
#[no_mangle]
pub fn discriminator_fn_item() -> u64 {
    // CHECK: ret i64 0
    const { discriminator_of_val(&item) }
}

// CHECK-LABEL: @discriminator_fn_item_coerced(
#[no_mangle]
pub fn discriminator_fn_item_coerced() -> u64 {
    // CHECK: ret i64 18983
    const { discriminator_of_val::<F0>(&(item as F0)) }
}
