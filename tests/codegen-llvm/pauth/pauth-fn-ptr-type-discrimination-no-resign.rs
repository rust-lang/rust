// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// A function pointer is signed with the discriminator of the function it points to, and
// reinterpreting it never resigns it. Transmute, `as` casts, union reads, reads through a cast
// pointer and memcpy are all plain bit copies. A round trip through another function pointer type,
// a raw pointer or an integer therefore needs nothing: the final call authenticates with the
// discriminator of `extern "C" fn()`, which is the one `f` was signed with.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::Option::{self, Some};
use minicore::copy_nonoverlapping;
use minicore::mem::{size_of, transmute};

type F0 = extern "C" fn();
type F1 = extern "C" fn(i32);

extern "C" fn f() {}
extern "C" fn g(_: i32) {}
extern "C" fn h() {}

union Pun {
    f0: F0,
    f1: F1,
}

// CHECK-LABEL: @via_transmute(
#[no_mangle]
pub fn via_transmute() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let q: F1 = unsafe { transmute(p) };
    let r: F0 = unsafe { transmute(q) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// Calling through the erased type authenticates with that type's discriminator. The pointer is
// still signed for `fn()`, so this traps at runtime, as calling `f` as `fn(i32)` is UB.
// CHECK-LABEL: @via_transmute_call_erased(
#[no_mangle]
pub fn via_transmute_call_erased() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let q: F1 = unsafe { transmute(p) };
    // DISC: call void {{.*}}(i32 1) {{.*}}[ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void {{.*}}(i32 1) {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    q(1);
}

// CHECK-LABEL: @via_union(
#[no_mangle]
pub fn via_union() {
    // CHECK-NOT: llvm.ptrauth.resign
    let u = Pun { f0: f };
    let q: F1 = unsafe { u.f1 };
    let u2 = Pun { f1: q };
    let r: F0 = unsafe { u2.f0 };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_pointer_read(
#[no_mangle]
pub fn via_pointer_read() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let q: F1 = unsafe { *(&p as *const F0 as *const F1) };
    let r: F0 = unsafe { *(&q as *const F1 as *const F0) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_memcpy(
#[no_mangle]
pub fn via_memcpy() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let mut q: F1 = g;
    let mut r: F0 = h;
    unsafe {
        copy_nonoverlapping(
            &p as *const F0 as *const u8,
            &mut q as *mut F1 as *mut u8,
            const { size_of::<F0>() },
        );
        copy_nonoverlapping(
            &q as *const F1 as *const u8,
            &mut r as *mut F0 as *mut u8,
            const { size_of::<F0>() },
        );
    }
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_as_cast(
#[no_mangle]
pub fn via_as_cast() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let raw = p as *const ();
    let r: F0 = unsafe { transmute(raw) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_raw_pointer(
#[no_mangle]
pub fn via_raw_pointer() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let raw: *const () = unsafe { transmute(p) };
    let r: F0 = unsafe { transmute(raw) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_integer(
#[no_mangle]
pub fn via_integer() {
    // CHECK-NOT: llvm.ptrauth.resign
    let p: F0 = f;
    let addr: usize = unsafe { transmute(p) };
    let r: F0 = unsafe { transmute(addr) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    r();
}

// CHECK-LABEL: @via_option(
#[no_mangle]
pub fn via_option() {
    // CHECK-NOT: llvm.ptrauth.resign
    // CHECK-NOT: ptrauth.null
    let p: Option<F0> = Some(f);
    let q: Option<F1> = unsafe { transmute(p) };
    let r: Option<F0> = unsafe { transmute(q) };
    // DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void {{.*}}() {{.*}}[ "ptrauth"(i32 0, i64 0) ]
    if let Some(r) = r {
        r();
    }
}

// The intrinsic is not even declared.
// CHECK-NOT: @llvm.ptrauth.resign
