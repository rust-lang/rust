// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included). Both compilers must generate identical
// values.
//
// Emulate NULL-able function argument with Option<FnPtr>. Make sure that Option<FnPtr> is treated
// as function pointer - encoded as P.
//
// Equivalent C sample:
//
// ```c
// #include <stdio.h>
//
// typedef int (*FnCallback)(int);
//
// int f_opt(FnCallback cb);
// int f_raw(FnCallback cb);
//
// int d_opt(void *ctx);
// int d_raw(void *ctx);
//
// int callback_i32(int);
//
// int (*T_OPT)(FnCallback) = f_opt;
// int (*T_RAW)(FnCallback) = f_raw;
//
// int (*D_OPT)(void *) = d_opt;
// int (*D_RAW)(void *) = d_raw;
// int main(void) {
//   /* function pointers */
//   T_OPT(callback_i32);
//   T_OPT(NULL);
//
//   T_RAW(callback_i32);
//
//   /* data pointers */
//   int x = 42;
//   D_OPT(&x);
//   D_OPT(NULL);
//
//   return 0;
// }
// ```

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

    fn g_opt(ctx: Option<*mut c_void>) -> i32;
    fn g_raw(ctx: *mut c_void) -> i32;

    fn callback_i32(x: i32) -> i32;
}

type FnOpt = unsafe extern "C" fn(Option<unsafe extern "C" fn(i32) -> i32>) -> i32;
type FnRaw = unsafe extern "C" fn(unsafe extern "C" fn(i32) -> i32) -> i32;
type DataOpt = unsafe extern "C" fn(Option<*mut c_void>) -> i32;
type DataRaw = unsafe extern "C" fn(*mut c_void) -> i32;

#[used]
// DISC: @{{.*}}T_OPT = constant ptr ptrauth (ptr @{{.*}}f_opt, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_OPT = constant ptr ptrauth (ptr @{{.*}}f_opt, i32 0), align 8
static T_OPT: FnOpt = f_opt;
#[used]
// DISC: @{{.*}}T_RAW = constant ptr ptrauth (ptr @{{.*}}f_raw, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_RAW = constant ptr ptrauth (ptr @{{.*}}f_raw, i32 0), align 8
static T_RAW: FnRaw = f_raw;

// DISC: @{{.*}}G_OPT = constant ptr ptrauth (ptr @{{.*}}g_opt, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}G_OPT = constant ptr ptrauth (ptr @{{.*}}g_opt, i32 0), align 8
#[used]
static G_OPT: DataOpt = g_opt;

// DISC: @{{.*}}G_RAW = constant ptr ptrauth (ptr @{{.*}}g_raw, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}G_RAW = constant ptr ptrauth (ptr @{{.*}}g_raw, i32 0), align 8
#[used]
static G_RAW: DataRaw = g_raw;
// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;

    unsafe {
        // Function pointers
        //DISC: call i32 ptrauth (ptr @f_opt, i32 0, i64 12410)(ptr ptrauth (ptr @callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        //NO_DISC: call i32 ptrauth (ptr @f_opt, i32 0)(ptr ptrauth (ptr @callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT(Some(callback_i32));
        //DISC: call i32 ptrauth (ptr @f_opt, i32 0, i64 12410)(ptr null) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        //NO_DISC: call i32 ptrauth (ptr @f_opt, i32 0)(ptr null) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT(None);
        // DISC: call i32 ptrauth (ptr @f_raw, i32 0, i64 12410)(ptr ptrauth (ptr @callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_raw, i32 0)(ptr ptrauth (ptr @callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_RAW(callback_i32);

        // Data pointers
        // DISC: call i32 ptrauth (ptr @g_opt, i32 0, i64 12410){{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @g_opt, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_OPT(Some((&mut x as *mut i32) as *mut c_void));
        // DISC: call i32 ptrauth (ptr @g_opt, i32 0, i64 12410){{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @g_opt, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_OPT(None);

        // DISC: call i32 ptrauth (ptr @g_raw, i32 0, i64 12410){{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @g_raw, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = G_RAW((&mut x as *mut i32) as *mut c_void);
    }
}
