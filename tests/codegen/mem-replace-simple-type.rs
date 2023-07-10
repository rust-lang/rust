// compile-flags: -O -C no-prepopulate-passes
// min-llvm-version: 15.0 (for opaque pointers)
// only-x86_64 (to not worry about usize differing)
// ignore-debug (the debug assertions get in the way)

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
// CHECK-LABEL: @replace_short_array(
pub fn replace_short_array(r: &mut [u32; 3], v: [u32; 3]) -> [u32; 3] {
    // CHECK-NOT: alloca
    // CHECK: %[[R:.+]] = load <3 x i32>, ptr %r, align 4
    // CHECK: store <3 x i32> %[[R]], ptr %result
    // CHECK: %[[V:.+]] = load <3 x i32>, ptr %v, align 4
    // CHECK: store <3 x i32> %[[V]], ptr %r
    std::mem::replace(r, v)
}
