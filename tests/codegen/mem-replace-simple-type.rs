//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-x86_64 (to not worry about usize differing)
//@ ignore-std-debug-assertions
// Reason: precondition checks make mem::replace not a candidate for MIR inlining

#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @replace_usize(
pub fn replace_usize(r: &mut usize, v: usize) -> usize {
    // CHECK-NOT: alloca
    // CHECK: %[[R:.+]] = load i64, ptr %r
    // CHECK: store i64 %v, ptr %r
    // CHECK: ret i64 %[[R]]
    std::mem::replace(r, v)
}

#[no_mangle]
// CHECK-LABEL: @replace_ref_str(
pub fn replace_ref_str<'a>(r: &mut &'a str, v: &'a str) -> &'a str {
    // CHECK-NOT: alloca
    // CHECK: %[[A:.+]] = load ptr
    // CHECK: %[[B:.+]] = load i64
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK: store ptr
    // CHECK: store i64
    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK: %[[P1:.+]] = insertvalue { ptr, i64 } poison, ptr %[[A]], 0
    // CHECK: %[[P2:.+]] = insertvalue { ptr, i64 } %[[P1]], i64 %[[B]], 1
    // CHECK: ret { ptr, i64 } %[[P2]]
    std::mem::replace(r, v)
}

#[no_mangle]
// CHECK-LABEL: @replace_short_array_3(
// CHECK-SAME: ptr{{.+}}sret{{.+}}%[[RET:.+]], ptr{{.+}}%r, ptr{{.+}}%v
pub fn replace_short_array_3(r: &mut [u32; 3], v: [u32; 3]) -> [u32; 3] {
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[RET]], ptr align 4 %r, i64 12, i1 false)
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %r, ptr align 4 %v, i64 12, i1 false)
    std::mem::replace(r, v)
}

#[no_mangle]
// CHECK-LABEL: @replace_short_array_4(
// CHECK-SAME: ptr{{.+}}sret{{.+}}%[[RET:.+]], ptr{{.+}}%r, ptr{{.+}}%v
pub fn replace_short_array_4(r: &mut [u32; 4], v: [u32; 4]) -> [u32; 4] {
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[RET]], ptr align 4 %r, i64 16, i1 false)
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %r, ptr align 4 %v, i64 16, i1 false)
    std::mem::replace(r, v)
}
