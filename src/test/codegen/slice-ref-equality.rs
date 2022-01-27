// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @is_zero_slice_short
#[no_mangle]
pub fn is_zero_slice_short(data: &[u8; 4]) -> bool {
    // CHECK: :
    // CHECK-NEXT: %[[PTR:.+]] = bitcast [4 x i8]* {{.+}} to i32*
    // CHECK-NEXT: %[[LOAD:.+]] = load i32, i32* %[[PTR]], align 1
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    &data[..] == [0; 4]
}

// CHECK-LABEL: @is_zero_array
#[no_mangle]
pub fn is_zero_array(data: &[u8; 4]) -> bool {
    // CHECK: start:
    // CHECK-NEXT: %[[PTR:.+]] = bitcast [4 x i8]* {{.+}} to i32*
    // CHECK-NEXT: %[[LOAD:.+]] = load i32, i32* %[[PTR]], align 1
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    *data == [0; 4]
}
