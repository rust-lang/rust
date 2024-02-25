//@ compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @array_copy_1_element
#[no_mangle]
pub fn array_copy_1_element(a: &[u8; 1], p: &mut [u8; 1]) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = load i8, ptr %a, align 1
    // CHECK: store i8 %[[TEMP]], ptr %p, align 1
    // CHECK: ret
    *p = *a;
}

// CHECK-LABEL: @array_copy_2_elements
#[no_mangle]
pub fn array_copy_2_elements(a: &[u8; 2], p: &mut [u8; 2]) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = load <2 x i8>, ptr %a, align 1
    // CHECK: store <2 x i8> %[[TEMP]], ptr %p, align 1
    // CHECK: ret
    *p = *a;
}

// CHECK-LABEL: @array_copy_4_elements
#[no_mangle]
pub fn array_copy_4_elements(a: &[u8; 4], p: &mut [u8; 4]) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = load <4 x i8>, ptr %a, align 1
    // CHECK: store <4 x i8> %[[TEMP]], ptr %p, align 1
    // CHECK: ret
    *p = *a;
}
