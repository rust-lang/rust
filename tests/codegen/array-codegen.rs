// compile-flags: -O -C no-prepopulate-passes
// min-llvm-version: 15.0 (for opaque pointers)

#![crate_type = "lib"]

// CHECK-LABEL: @array_load
#[no_mangle]
pub fn array_load(a: &[u8; 4]) -> [u8; 4] {
    // CHECK: %0 = alloca [4 x i8], align 1
    // CHECK: %[[TEMP1:.+]] = load <4 x i8>, ptr %a, align 1
    // CHECK: store <4 x i8> %[[TEMP1]], ptr %0, align 1
    // CHECK: %[[TEMP2:.+]] = load i32, ptr %0, align 1
    // CHECK: ret i32 %[[TEMP2]]
    *a
}

// CHECK-LABEL: @array_store
#[no_mangle]
pub fn array_store(a: [u8; 4], p: &mut [u8; 4]) {
    // CHECK: %a = alloca [4 x i8]
    // CHECK: %[[TEMP:.+]] = load <4 x i8>, ptr %a, align 1
    // CHECK-NEXT: store <4 x i8> %[[TEMP]], ptr %p, align 1
    *p = a;
}

// CHECK-LABEL: @array_copy
#[no_mangle]
pub fn array_copy(a: &[u8; 4], p: &mut [u8; 4]) {
    // CHECK: %[[LOCAL:.+]] = alloca [4 x i8], align 1
    // CHECK: %[[TEMP1:.+]] = load <4 x i8>, ptr %a, align 1
    // CHECK: store <4 x i8> %[[TEMP1]], ptr %[[LOCAL]], align 1
    // CHECK: %[[TEMP2:.+]] = load <4 x i8>, ptr %[[LOCAL]], align 1
    // CHECK: store <4 x i8> %[[TEMP2]], ptr %p, align 1
    *p = *a;
}
