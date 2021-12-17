// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// #71602 reported a simple array comparison just generating a loop.
// This was originally fixed by ensuring it generates a single bcmp,
// but we now generate it as a load+icmp instead. `is_zero_slice` was
// tweaked to still test the case of comparison against a slice,
// and `is_zero_array` tests the new array-specific behaviour.
// The optimization was then extended to short slice-to-array comparisons,
// so the first test here now has a long slice to still get the bcmp.

// CHECK-LABEL: @is_zero_slice_long
#[no_mangle]
pub fn is_zero_slice_long(data: &[u8; 456]) -> bool {
    // CHECK: :
    // CHECK-NEXT: %{{.+}} = getelementptr {{.+}}
    // CHECK-NEXT: %[[BCMP:.+]] = tail call i32 @{{bcmp|memcmp}}({{.+}})
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[BCMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    &data[..] == [0; 456]
}

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
