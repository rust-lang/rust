// Verifies that the `alloc_token_infer` intrinsic (i.e., the Rust equivalent of the Clang
// `__builtin_infer_alloc_token` builtin) queries the `llvm.alloc.token.id` intrinsic for the
// same `!{<type-name>, <contains-pointer>}` metadata used for `!alloc_token` metadata.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::alloc_token_infer;

// A type not containing pointers.
pub struct NoPointers(u64);

// A type containing pointers.
pub struct Pointers(*const u8);

// CHECK-LABEL: {{.*}}foo{{.*}}(
pub fn foo() -> (usize, usize) {
    // CHECK: insertvalue {{.*}} i64 0, 0
    // CHECK: insertvalue {{.*}} i64 1, 1
    (unsafe { alloc_token_infer::<NoPointers>() }, unsafe { alloc_token_infer::<Pointers>() })
}

// CHECK: declare {{.*}} @llvm.alloc.token.id{{.*}}(metadata)
