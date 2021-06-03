// compile-flags: -O
// only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @array_eq_value
#[no_mangle]
pub fn array_eq_value(a: [u16; 6], b: [u16; 6]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %2 = icmp eq i96 %0, %1
    // CHECK-NEXT: ret i1 %2
    a == b
}

// CHECK-LABEL: @array_eq_ref
#[no_mangle]
pub fn array_eq_ref(a: &[u16; 6], b: &[u16; 6]) -> bool {
    // CHECK: start:
    // CHECK: load i96, i96* %{{.+}}, align 2
    // CHECK: load i96, i96* %{{.+}}, align 2
    // CHECK: icmp eq i96
    // CHECK-NEXT: ret
    a == b
}

// CHECK-LABEL: @array_eq_value_still_passed_by_pointer
#[no_mangle]
pub fn array_eq_value_still_passed_by_pointer(a: [u16; 9], b: [u16; 9]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* nonnull dereferenceable(18) %{{.+}}, i8* nonnull dereferenceable(18) %{{.+}}, i64 18)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_long
#[no_mangle]
pub fn array_eq_long(a: &[u16; 1234], b: &[u16; 1234]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* nonnull dereferenceable(2468) %{{.+}}, i8* nonnull dereferenceable(2468) %{{.+}}, i64 2468)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_zero(i128 %0)
#[no_mangle]
pub fn array_eq_zero(x: [u16; 8]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i128 %0, 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 8]
}
