//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK-LABEL: @array_load
#[no_mangle]
pub fn array_load(a: &[u8; 4]) -> [u8; 4] {
    // CHECK-NOT: alloca
    // CHECK: %[[ALLOCA:.+]] = alloca [4 x i8], align 1
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %[[ALLOCA]], ptr align 1 %a, {{.+}} 4, i1 false)
    // CHECK: %[[TEMP:.+]] = load i32, ptr %[[ALLOCA]], align 1
    // CHECK: ret i32 %[[TEMP]]
    *a
}

// CHECK-LABEL: @array_store
#[no_mangle]
pub fn array_store(a: [u8; 4], p: &mut [u8; 4]) {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [4 x i8], [[TEMPALIGN:align [0-9]+]]
    // CHECK-NOT: alloca
    // CHECK: %a = alloca [4 x i8]
    // CHECK-NOT: alloca
    // store i32 %0, ptr %[[TEMP]]
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %a, ptr [[TEMPALIGN]] %[[TEMP]], {{.+}} 4, i1 false)
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %p, ptr align 1 %a, {{.+}} 4, i1 false)
    *p = a;
}

// CHECK-LABEL: @array_copy
#[no_mangle]
pub fn array_copy(a: &[u8; 4], p: &mut [u8; 4]) {
    // CHECK-NOT: alloca
    // CHECK: %[[LOCAL:.+]] = alloca [4 x i8], align 1
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %[[LOCAL]], ptr align 1 %a, {{.+}} 4, i1 false)
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %p, ptr align 1 %[[LOCAL]], {{.+}} 4, i1 false)
    *p = *a;
}

// CHECK-LABEL: @array_copy_1_element
#[no_mangle]
pub fn array_copy_1_element(a: &[u8; 1], p: &mut [u8; 1]) {
    // CHECK-NOT: alloca
    // CHECK: %[[LOCAL:.+]] = alloca [1 x i8], align 1
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %[[LOCAL]], ptr align 1 %a, {{.+}} 1, i1 false)
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %p, ptr align 1 %[[LOCAL]], {{.+}} 1, i1 false)
    *p = *a;
}

// CHECK-LABEL: @array_copy_2_elements
#[no_mangle]
pub fn array_copy_2_elements(a: &[u8; 2], p: &mut [u8; 2]) {
    // CHECK-NOT: alloca
    // CHECK: %[[LOCAL:.+]] = alloca [2 x i8], align 1
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %[[LOCAL]], ptr align 1 %a, {{.+}} 2, i1 false)
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align 1 %p, ptr align 1 %[[LOCAL]], {{.+}} 2, i1 false)
    *p = *a;
}
