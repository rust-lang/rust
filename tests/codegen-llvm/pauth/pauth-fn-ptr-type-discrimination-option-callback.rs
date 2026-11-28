// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Nullable pointer arguments emulated with `Option`. A two-variant enum is encoded as its payload
// only when it has the payload's layout:
// * `Option<fn>` and `Option<&T>` use the payload's null niche, so they are ABI-compatible with
//   the bare pointer and encode as `P`.
// * `Option<*mut T>` needs a tag, because null is a valid raw pointer, so it must not encode as
//   `P`.
//
// Discriminators:
// 18786: "Fi6OptionE"
// 12410: "FiPE"
// 2981:  "FiiE"

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::Option::{None, Some};
use minicore::{Option, c_void};

extern "C" {
    fn f_opt(cb: Option<unsafe extern "C" fn(i32) -> i32>) -> i32;
    fn f_raw(cb: unsafe extern "C" fn(i32) -> i32) -> i32;

    fn r_opt(ctx: Option<&i32>) -> i32;

    fn g_opt(ctx: Option<*mut c_void>) -> i32;
    fn g_raw(ctx: *mut c_void) -> i32;

    fn callback_i32(x: i32) -> i32;
}

type FnOpt = unsafe extern "C" fn(Option<unsafe extern "C" fn(i32) -> i32>) -> i32;
type FnRaw = unsafe extern "C" fn(unsafe extern "C" fn(i32) -> i32) -> i32;
type RefOpt = unsafe extern "C" fn(Option<&i32>) -> i32;
type DataOpt = unsafe extern "C" fn(Option<*mut c_void>) -> i32;
type DataRaw = unsafe extern "C" fn(*mut c_void) -> i32;

// The statics may be emitted in any order.
// DISC-DAG: @{{.*}}T_OPT = constant ptr ptrauth (ptr @{{.*}}f_opt, i32 0, i64 12410), align 8
// DISC-DAG: @{{.*}}T_RAW = constant ptr ptrauth (ptr @{{.*}}f_raw, i32 0, i64 12410), align 8
// DISC-DAG: @{{.*}}G_OPT = constant ptr ptrauth (ptr @{{.*}}g_opt, i32 0, i64 18786), align 8
// DISC-DAG: @{{.*}}G_RAW = constant ptr ptrauth (ptr @{{.*}}g_raw, i32 0, i64 12410), align 8
// NO_DISC-DAG: @{{.*}}T_OPT = constant ptr ptrauth (ptr @{{.*}}f_opt, i32 0), align 8
// NO_DISC-DAG: @{{.*}}T_RAW = constant ptr ptrauth (ptr @{{.*}}f_raw, i32 0), align 8
// NO_DISC-DAG: @{{.*}}G_OPT = constant ptr ptrauth (ptr @{{.*}}g_opt, i32 0), align 8
// NO_DISC-DAG: @{{.*}}G_RAW = constant ptr ptrauth (ptr @{{.*}}g_raw, i32 0), align 8
#[used]
static T_OPT: FnOpt = f_opt;
#[used]
static T_RAW: FnRaw = f_raw;
#[used]
static G_OPT: DataOpt = g_opt;
#[used]
static G_RAW: DataRaw = g_raw;

// `Option<&i32>` makes this a higher-ranked function pointer type, which does not implement
// `Sync` in minicore, so it is a `const` rather than a `static`.
const R_OPT: RefOpt = r_opt;

// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;

    unsafe {
        // Function pointers.
        // DISC: call i32 ptrauth (ptr @f_opt, i32 0, i64 12410)(ptr ptrauth (ptr @callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_opt, i32 0)(ptr ptrauth (ptr @callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT(Some(callback_i32));
        // DISC: call i32 ptrauth (ptr @f_opt, i32 0, i64 12410)(ptr null) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_opt, i32 0)(ptr null) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT(None);
        // DISC: call i32 ptrauth (ptr @f_raw, i32 0, i64 12410)(ptr ptrauth (ptr @callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_raw, i32 0)(ptr ptrauth (ptr @callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_RAW(callback_i32);

        // References.
        // DISC: call i32 ptrauth (ptr @r_opt, i32 0, i64 12410)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @r_opt, i32 0)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = R_OPT(Some(&x));

        // Raw pointers.
        // DISC: call i32 ptrauth (ptr @g_opt, i32 0, i64 18786){{.*}} [ "ptrauth"(i32 0, i64 18786) ]
        // NO_DISC: call i32 ptrauth (ptr @g_opt, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_OPT(Some((&mut x as *mut i32) as *mut c_void));
        // DISC: call i32 ptrauth (ptr @g_opt, i32 0, i64 18786){{.*}} [ "ptrauth"(i32 0, i64 18786) ]
        // NO_DISC: call i32 ptrauth (ptr @g_opt, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_OPT(None);
        // DISC: call i32 ptrauth (ptr @g_raw, i32 0, i64 12410){{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @g_raw, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_RAW((&mut x as *mut i32) as *mut c_void);
    }
}
