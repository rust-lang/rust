// compile-flags: -O

#![crate_type = "lib"]

// CHECK-LABEL: @array_clone
#[no_mangle]
pub fn array_clone(a: &[u8; 2]) -> [u8; 2] {
    // CHECK: getelementptr
    // CHECK-NEXT: load i8
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: load i8
    // CHECK: ret [2 x i8]
    a.clone()
}

// CHECK-LABEL: @array_clone_big
#[no_mangle]
pub fn array_clone_big(a: &[u8; 16]) -> [u8; 16] {
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: getelementptr inbounds [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: insertvalue [16 x i8]
    // CHECK: ret [16 x i8]
    a.clone()
}
