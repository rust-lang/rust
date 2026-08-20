// Verifies that allocation calls carrying `!alloc_token` metadata are rewritten by the
// `AllocTokenPass` to the token-enabled allocator interface (i.e., prefixed with
// `__alloc_token_`, with the token identifier appended as the last argument).
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]
#![feature(rustc_attrs)]

unsafe extern "Rust" {
    #[rustc_allocator]
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

// CHECK-LABEL: define{{.*}}3foo
// CHECK: call{{.*}}@__alloc_token_{{.*}}__rust_alloc({{.*}}, i64 1)
pub unsafe fn foo(size: usize, align: usize) -> *mut u8 {
    unsafe { __rust_alloc(size, align) }
}
