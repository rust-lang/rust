// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0
//
// Make sure that a function pointer materialized from a `const` item carries the same discriminator
// as the identical signature materialized from a `static`.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;

extern "C" {
    fn f(ctx: *mut i32) -> i32;
}

type FnPtr = unsafe extern "C" fn(*mut i32) -> i32;

#[used]
// DISC: @{{.*}}T_STATIC = constant ptr ptrauth (ptr @{{.*}}f, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_STATIC = constant ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static T_STATIC: FnPtr = f;

// `const`, not `static`: this is the exact case the fix addresses. A bare standalone function
// pointer constant has no separate global of its own - it's inlined as an LLVM constant operand
// directly at each use site, which is why the check below lives at the call site rather than on
// a symbol (unlike T_STATIC above).
const T_CONST: FnPtr = f;

// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;
    unsafe {
        // DISC: call i32 ptrauth (ptr @f, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_STATIC(&mut x as *mut i32);

        // DISC: call i32 ptrauth (ptr @f, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_CONST(&mut x as *mut i32);
    }
}
