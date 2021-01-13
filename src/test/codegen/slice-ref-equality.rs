// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// #71602: check that slice equality just generates a single bcmp

// CHECK-LABEL: @is_zero_slice
#[no_mangle]
pub fn is_zero_slice(data: &[u8; 4]) -> bool {
    // CHECK: start:
    // CHECK-NEXT: %{{.+}} = getelementptr {{.+}}
    // CHECK-NEXT: %[[BCMP:.+]] = tail call i32 @{{bcmp|memcmp}}({{.+}})
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[BCMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    *data == [0; 4]
}
