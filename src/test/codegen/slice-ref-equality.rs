// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @is_zero_slice_short
#[no_mangle]
pub fn is_zero_slice_short(data: &[u8; 4]) -> bool {
    // CHECK: getelementptr [4 x i8], [4 x i8]*
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    &data[..] == [0; 4]
}

// CHECK-LABEL: @is_zero_array
#[no_mangle]
pub fn is_zero_array(data: &[u8; 4]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: getelementptr [4 x i8], [4 x i8]*
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    *data == [0; 4]
}
