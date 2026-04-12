// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH O0_NO_PAUTH O3_NO_PAUTH

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

// O0_PAUTH: @{{[0-9A-Za-z_]+}}FUNCTION_PTR_DECL = constant ptr ptrauth (ptr @copy_file_range, i32 0)
// O0_PAUTH: declare i64 @copy_file_range({{.*}})
// O3_PAUTH: @{{[0-9A-Za-z_]+}}FUNCTION_PTR_DECL = constant ptr ptrauth (ptr @copy_file_range, i32 0)
// O3_PAUTH: declare {{.*}} i64 @copy_file_range({{.*}})
//
// O0_NO_PAUTH-NOT: ptr ptrauth
// O3_NO_PAUTH-NOT: ptr ptrauth
extern "C" {
    #[link_name = "copy_file_range"]
    #[linkage = "extern_weak"]
    fn copy_file_range(
        fd_in: i32,
        off_in: *mut i64,
        fd_out: i32,
        off_out: *mut i64,
        len: i64,
        flags: i32,
    ) -> i64;
}

#[used]
static FUNCTION_PTR_DECL: unsafe extern "C" fn(i32, *mut i64, i32, *mut i64, i64, i32) -> i64 =
    copy_file_range;

// O0_PAUTH: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
// O3_PAUTH: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}

// O0_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
// O3_NO_PAUTH-NOT: !{{[0-9]+}} = !{i32 1, !"ptrauth-sign-personality", i32 1}
