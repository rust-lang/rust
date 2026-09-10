// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Regression test for `canonicalize_c_type`
// (compiler/rustc_middle/src/ptrauth/discriminator.rs).
//
// `Option<&T>` is ABI-compatible with `&T` because references have a null pointer niche, so it must
// receive the same discriminator as a pointer.
//
// `Option<*mut T>` is not ABI-compatible with a bare pointer: raw pointers do not have a null niche
// because null is a valid pointer value. It must therefore not receive the pointer discriminator.
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
use minicore::Option;
use minicore::Option::Some;

extern "C" {
    fn f_ptr(ctx: *mut i32) -> i32;
    fn f_int(x: i32) -> i32;
    fn f_ref(ctx: &i32) -> i32;
    fn f_opt_ref(ctx: Option<&i32>) -> i32;
    fn f_opt_raw(ctx: Option<*mut i32>) -> i32;
}

type FnPtr = unsafe extern "C" fn(*mut i32) -> i32;
type FnInt = unsafe extern "C" fn(i32) -> i32;
type FnRef = unsafe extern "C" fn(&i32) -> i32;
type FnOptRef = unsafe extern "C" fn(Option<&i32>) -> i32;
type FnOptRaw = unsafe extern "C" fn(Option<*mut i32>) -> i32;

#[used]
// Baseline: bare pointer argument.
// DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f_ptr, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @{{.*}}f_ptr, i32 0), align 8
static T_PTR: FnPtr = f_ptr;

#[used]
// Baseline: bare scalar-int argument.
// DISC: @{{.*}}T_INT = constant ptr ptrauth (ptr @{{.*}}f_int, i32 0, i64 2981), align 8
// NO_DISC: @{{.*}}T_INT = constant ptr ptrauth (ptr @{{.*}}f_int, i32 0), align 8
static T_INT: FnInt = f_int;

const T_REF: FnRef = f_ref;

// Option<&T> is niche-optimized (ABI-identical to &T), so it must match T_PTR/T_REF - NOT T_INT.
// Under the pre-fix code this failed: Option<&T> wasn't canonicalized at all and fell through
// to the enum path, so it wrongly matched T_INT instead.
const T_OPT_REF: FnOptRef = f_opt_ref;

#[used]
// Option<*mut T> is NOT niche-optimized, so canonicalize_c_type must NOT unwrap it to a pointer -
// it must NOT match T_PTR/T_REF's [[PTR_DISC]] bucket. Under the pre-fix code this failed:
// Option<*mut T> was wrongly canonicalized straight to Pointer, so it wrongly matched T_PTR.
//
// Which non-pointer bucket it actually lands in is NOT this file's concern - that's determined
// by a separate piece of logic (the enum payload-collapse guard in `to_clang_disc_ty`), tested
// independently in pauth-fn-ptr-type-discrimination-enum-payload.rs. Asserting a specific value
// here would wrongly couple this test to that other fix.
// DISC-NOT: @{{.*}}T_OPT_RAW = constant ptr ptrauth (ptr @{{.*}}f_opt_raw, i32 0, i64 12410), align 8
// DISC: @{{.*}}T_OPT_RAW = constant ptr ptrauth (ptr @{{.*}}f_opt_raw, i32 0, i64 18786), align 8
// NO_DISC: @{{.*}}T_OPT_RAW = constant ptr ptrauth (ptr @{{.*}}f_opt_raw, i32 0), align 8
static T_OPT_RAW: FnOptRaw = f_opt_raw;

// CHECK-LABEL: main
pub fn main() {
    let mut x = 42i32;
    unsafe {
        // DISC: call i32 ptrauth (ptr @f_ptr, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_ptr, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_PTR(&mut x as *mut i32);

        // DISC: call i32 ptrauth (ptr @f_int, i32 0, i64 2981)(i32 %{{.*}}) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 ptrauth (ptr @f_int, i32 0)(i32 %{{.*}}) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_INT(x);

        // Call-site bundles must carry the same discriminators as the signing above.
        // DISC: call i32 ptrauth (ptr @f_ref, i32 0, i64 12410)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_ref, i32 0)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_REF(&x);

        // DISC: call i32 ptrauth (ptr @f_opt_ref, i32 0, i64 12410)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_opt_ref, i32 0)(ptr align 4 %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT_REF(Some(&x));

        // DISC: call i32 ptrauth (ptr @f_opt_raw, i32 0, i64 18786){{.*}} [ "ptrauth"(i32 0, i64 18786) ]
        // NO_DISC: call i32 ptrauth (ptr @f_opt_raw, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_OPT_RAW(Some(&mut x as *mut i32));
    }
}
