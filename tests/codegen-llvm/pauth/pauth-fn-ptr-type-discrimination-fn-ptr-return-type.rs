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
// Test generation of function-pointer type discriminators for functions returning a function
// pointer themselves.
//
// Equivalent C sample:
//
// ```c
// int (*f_ret_fp(int x))(int);
//
// int (*(*T_RET_FP)(int))(int) = f_ret_fp;
//
// int main(void) {
//   int (*cb)(int) = T_RET_FP(123);
//   return cb(456);
// }
// ```

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;

extern "C" {
    fn f_ret_fp(x: i32) -> extern "C" fn(i32) -> i32;
}

type FnCallback = extern "C" fn(i32) -> i32;
type FnRetFp = unsafe extern "C" fn(i32) -> FnCallback;

#[used]
// discriminator: 32957 (0x80BD), encoding: FPiE
// DISC: @{{.*}}T_RET_FP = constant ptr ptrauth (ptr @f_ret_fp, i32 0, i64 32957), align 8
// NO_DISC: @{{.*}}T_RET_FP = constant ptr ptrauth (ptr @f_ret_fp, i32 0), align 8
static T_RET_FP: FnRetFp = f_ret_fp;

pub fn main() {
    unsafe {
        // discriminator: 32957 (0x80BD), encoding: FPiE
        // DISC: [[CB:%.*]] = call ptr ptrauth (ptr @f_ret_fp, i32 0, i64 32957)(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 32957) ]
        // NO_DISC: [[CB:%.*]] = call ptr ptrauth (ptr @f_ret_fp, i32 0)(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let cb = T_RET_FP(123);
        // discriminator: 2981 (0x0BA5), encoding: FiiE
        // DISC: call i32 [[CB]](i32 456) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 [[CB]](i32 456) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = cb(456);
    }
}
