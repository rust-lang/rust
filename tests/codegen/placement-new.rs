//@ compile-flags: -O
#![crate_type = "lib"]

// Test to check that types with "complex" destructors, but trivial `Default` impls
// are constructed directly into the allocation in `Box::default` and `Arc::default`.

use std::sync::Arc;

// CHECK-LABEL: @box_default_inplace
#[no_mangle]
pub fn box_default_inplace() -> Box<(String, String)> {
    // CHECK-NOT: alloca
    // CHECK: [[BOX:%.*]] = {{.*}}call {{.*}}__rust_alloc(
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: ret ptr [[BOX]]
    Box::default()
}

// CHECK-LABEL: @arc_default_inplace
#[no_mangle]
pub fn arc_default_inplace() -> Arc<(String, String)> {
    // CHECK-NOT: alloca
    // CHECK: [[ARC:%.*]] = {{.*}}call {{.*}}__rust_alloc(
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: ret ptr [[ARC]]
    Arc::default()
}
