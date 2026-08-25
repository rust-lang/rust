//@ compile-flags: -Copt-level=3
//@ compile-flags: -Zmerge-functions=disabled
#![crate_type = "lib"]

// Test to check that types with "complex" destructors, but trivial `Default` impls
// are constructed directly into the allocation in `Box::default` and `Arc::default`.

use std::rc::Rc;
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

// CHECK-LABEL: @rc_default_inplace
#[no_mangle]
pub fn rc_default_inplace() -> Rc<(String, String)> {
    // CHECK-NOT: alloca
    // CHECK: [[RC:%.*]] = {{.*}}call {{.*}}__rust_alloc(
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: ret ptr [[RC]]
    Rc::default()
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
