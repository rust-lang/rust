// Verifies that `!alloc_token` metadata is emitted for calls to allocation functions from typed
// allocation paths (i.e., functions annotated with the `#[rustc_alloc_token_hint]` attribute).
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]
#![feature(rustc_attrs)]

use std::mem::{align_of, size_of};

unsafe extern "Rust" {
    #[rustc_allocator]
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

#[rustc_alloc_token_hint]
unsafe fn alloc<T>() -> *mut u8 {
    unsafe { __rust_alloc(size_of::<T>(), align_of::<T>()) }
}

// A type not containing pointers.
pub struct NoPointers(u64);

// A type containing pointers.
pub struct Pointers(*const u8);

pub unsafe fn foo() -> (*mut u8, *mut u8) {
    (unsafe { alloc::<NoPointers>() }, unsafe { alloc::<Pointers>() })
}

// CHECK-DAG: !{{[0-9]+}} = !{!"{{.*}}NoPointers{{.*}}", i1 false}
// CHECK-DAG: !{{[0-9]+}} = !{!"{{.*}}Pointers{{.*}}", i1 true}
