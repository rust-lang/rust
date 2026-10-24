// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0
//
// Regression test for `#[repr(transparent)]` handling in `canonicalize_c_type`
// (compiler/rustc_middle/src/ptrauth/discriminator.rs).
//
// `#[repr(transparent)]` is Rust's explicit ABI guarantee that a wrapper struct has the exact size,
// alignment, and calling convention of its one significant (non-1-ZST) field. Clang's own encoder
// strips typedefs before encoding (`QT.getCanonicalType()`).
//
// Discriminators:
// 12410: "FiPE"
// 62802: "Fi14NotTransparentE"

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;
use minicore::Option;
use minicore::Option::Some;

#[repr(transparent)]
pub struct Handle(*mut i32);

// A same-shaped wrapper WITHOUT `#[repr(transparent)]` this must NOT collapse to the pointer bucket
pub struct NotTransparent(*mut i32);

pub struct Marker;

// A 1-ZST filler field placed before the significant field, to actually exercise `non_1zst_field`'s
// real search rather than something that would coincidentally also pass for a plain single-field
// wrapper.
#[repr(transparent)]
pub struct HandleWithMarker(Marker, *mut i32);

// Composition: peeling the transparent wrapper must happen before the Option<&T> rule can see
// what's inside it. Exercises the fixpoint loop in canonicalize_c_type.
#[repr(transparent)]
pub struct OptRefWrapper<'a>(Option<&'a i32>);

extern "C" {
    fn f_ptr(x: *mut i32) -> i32;
    fn f_handle(x: Handle) -> i32;
    fn f_not_transparent(x: NotTransparent) -> i32;
    fn f_handle_with_marker(x: HandleWithMarker) -> i32;
    fn f_opt_ref_wrapped(x: OptRefWrapper) -> i32;
}

type FnPtr = unsafe extern "C" fn(*mut i32) -> i32;
type FnHandle = unsafe extern "C" fn(Handle) -> i32;
type FnNotTransparent = unsafe extern "C" fn(NotTransparent) -> i32;
type FnHandleWithMarker = unsafe extern "C" fn(HandleWithMarker) -> i32;
type FnOptRefWrapped = unsafe extern "C" fn(OptRefWrapper) -> i32;

#[used]
// Baseline: bare pointer argument.
// DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f_ptr, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f_ptr, i32 0), align 8
static T_PTR: FnPtr = f_ptr;

#[used]
// DISC: @{{.*}}T_HANDLE = constant ptr ptrauth (ptr @{{.*}}f_handle, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_HANDLE = constant ptr ptrauth (ptr @{{.*}}f_handle, i32 0), align 8
static T_HANDLE: FnHandle = f_handle;

#[used]
// DISC-NOT: @{{.*}}T_NOT_TRANSPARENT = constant ptr ptrauth (ptr @{{.*}}f_not_transparent, i32 0, i64 12410), align 8
// DISC: @{{.*}}T_NOT_TRANSPARENT = constant ptr ptrauth (ptr @{{.*}}f_not_transparent, i32 0, i64 62802), align 8
// NO_DISC: @{{.*}}T_NOT_TRANSPARENT = constant ptr ptrauth (ptr @{{.*}}f_not_transparent, i32 0), align 8
static T_NOT_TRANSPARENT: FnNotTransparent = f_not_transparent;

#[used]
// DISC: @{{.*}}T_HANDLE_WITH_MARKER = constant ptr ptrauth (ptr @{{.*}}f_handle_with_marker, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_HANDLE_WITH_MARKER = constant ptr ptrauth (ptr @{{.*}}f_handle_with_marker, i32 0), align 8
static T_HANDLE_WITH_MARKER: FnHandleWithMarker = f_handle_with_marker;

const T_OPT_REF_WRAPPED: FnOptRefWrapped = f_opt_ref_wrapped;

// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;
    unsafe {
        // DISC: call i32 ptrauth (ptr @f_ptr, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_ptr, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_PTR(&mut x as *mut i32);

        // DISC: call i32 ptrauth (ptr @f_handle, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_handle, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_HANDLE(Handle(&mut x as *mut i32));

        // DISC: call i32 ptrauth (ptr @f_not_transparent, i32 0, i64 62802)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 62802) ]
        // NO_DISC: call i32 ptrauth (ptr @f_not_transparent, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_NOT_TRANSPARENT(NotTransparent(&mut x as *mut i32));

        // DISC: call i32 ptrauth (ptr @f_handle_with_marker, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_handle_with_marker, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_HANDLE_WITH_MARKER(HandleWithMarker(Marker, &mut x as *mut i32));

        // DISC: call i32 ptrauth (ptr @f_opt_ref_wrapped, i32 0, i64 12410)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_opt_ref_wrapped, i32 0)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT_REF_WRAPPED(OptRefWrapper(Some(&x)));
    }
}
