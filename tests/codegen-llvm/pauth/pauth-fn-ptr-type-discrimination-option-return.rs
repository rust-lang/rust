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
// Test generation of function-pointer type discriminators for optional returns.
//
// Equivalent C sample:
//
// ```c
// typedef int (*FnCallback)(int);
// FnCallback f_ret_option(void);
// FnCallback (*T_RET_OPTION)(void) = f_ret_option;
//
// int main(void) {
//   FnCallback cb = T_RET_OPTION();
//
//   if (cb)
//     cb(123);
// }
// ```

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::Option;
use minicore::Option::{None, Some};

extern "C" {
    fn f_ret_option() -> Option<extern "C" fn(i32) -> i32>;
}

type FnRetOption = unsafe extern "C" fn() -> Option<extern "C" fn(i32) -> i32>;

#[used]
// DISC: @{{.*}}T_RET_OPTION = constant ptr ptrauth (ptr @{{.*}}f_ret_option, i32 0, i64 34128), align 8
// NO_DISC: @{{.*}}T_RET_OPTION = constant ptr ptrauth (ptr @{{.*}}f_ret_option, i32 0), align 8
static T_RET_OPTION: FnRetOption = f_ret_option;

pub fn main() {
    unsafe {
        // DISC: call ptr ptrauth (ptr @f_ret_option, i32 0, i64 34128)() {{.*}} [ "ptrauth"(i32 0, i64 34128) ]
        // NO_DISC: call ptr ptrauth (ptr @f_ret_option, i32 0)() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        if let Some(cb) = T_RET_OPTION() {
            // DISC: call i32 %cb(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
            // NO_DISC: call i32 %cb(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
            let _ = cb(123);
        }
    }
}
