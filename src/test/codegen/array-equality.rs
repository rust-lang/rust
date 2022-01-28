// compile-flags: -O
// only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @array_eq_ref
#[no_mangle]
pub fn array_eq_ref(a: &[u16; 6], b: &[u16; 6]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast [6 x i16]
    // CHECK-NEXT: bitcast [6 x i16]
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(12) %{{.+}}, i8* {{.*}} dereferenceable(12) %{{.+}}, i64 12)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_value_still_passed_by_pointer
#[no_mangle]
pub fn array_eq_value_still_passed_by_pointer(a: [u16; 9], b: [u16; 9]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast [9 x i16]
    // CHECK-NEXT: bitcast [9 x i16]
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(18) %{{.+}}, i8* {{.*}} dereferenceable(18) %{{.+}}, i64 18)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_long
#[no_mangle]
pub fn array_eq_long(a: &[u16; 1234], b: &[u16; 1234]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast [1234 x i16]
    // CHECK-NEXT: bitcast [1234 x i16]
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(2468) %{{.+}}, i8* {{.*}} dereferenceable(2468) %{{.+}}, i64 2468)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_zero
#[no_mangle]
pub fn array_eq_zero(x: [u16; 8]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast [8 x i16]
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(16) %{{.+}}, i8* {{.*}} dereferenceable(16) {{.+}}, i64 16)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 8]
}
