// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH O0_NO_PAUTH O3_NO_PAUTH
//@ add-minicore

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3
//@ [O0_NO_PAUTH] needs-llvm-components: aarch64
//@ [O0_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=0
//@ [O3_NO_PAUTH] needs-llvm-components: aarch64
//@ [O3_NO_PAUTH] compile-flags: --target=aarch64-unknown-linux-gnu -C opt-level=3

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core)]
#![feature(linkage)]

extern crate minicore;
use minicore::*;

// O0_PAUTH: @{{[0-9A-Za-z_]+}}FUNCTION_PTR_DECL = constant ptr ptrauth (ptr @extern_weak_fn, i32 0)
// O0_PAUTH: declare i64 @extern_weak_fn({{.*}})
// O3_PAUTH: @{{[0-9A-Za-z_]+}}FUNCTION_PTR_DECL = constant ptr ptrauth (ptr @extern_weak_fn, i32 0)
// O3_PAUTH: declare {{.*}} i64 @extern_weak_fn({{.*}})
//
// O0_NO_PAUTH-NOT: ptr ptrauth
// O3_NO_PAUTH-NOT: ptr ptrauth
extern "C" {
    #[link_name = "extern_weak_fn"]
    #[linkage = "extern_weak"]
    fn extern_weak_fn() -> i64;
}

#[used]
static FUNCTION_PTR_DECL: unsafe extern "C" fn() -> i64 = extern_weak_fn;
