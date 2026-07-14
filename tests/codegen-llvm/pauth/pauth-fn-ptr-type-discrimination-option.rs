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
// Test generation of function-pointer type discriminators for optional variables.
//
// Equivalent C sample:
//
// ```c
// extern void f(int);
// void (*test_constant_null)(int) = 0;
// void (*test_constant_non_null)(int) = f;
// ```

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::Option;
use minicore::Option::{None, Some};

extern "C" fn f(_: i32) {}

// Rust function pointers are no-nullable, so this can not be expressed:
// void (*test_constant_null)(int) = 0;
// Use Option<TestConstantNullTy> instead.
type TestConstantNullTy = unsafe extern "C" fn(i32);

#[used]
// DISC: @{{.*}}TEST_CONSTANT_NON_NULL = constant ptr ptrauth (ptr @{{.*}}f, i32 0, i64 2712), align 8
// NO_DISC: @{{.*}}TEST_CONSTANT_NON_NULL = constant ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static TEST_CONSTANT_NON_NULL: Option<TestConstantNullTy> = Some(f);
#[used]
// CHECK: @{{.*}}TEST_CONSTANT_NULL = constant {{.*}} zeroinitializer, align 8
static TEST_CONSTANT_NULL: Option<TestConstantNullTy> = None;
