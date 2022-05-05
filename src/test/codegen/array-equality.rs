// compile-flags: -O
// only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @array_eq_value
#[no_mangle]
pub fn array_eq_value(a: [u16; 3], b: [u16; 3]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %2 = icmp eq i48 %0, %1
    // CHECK-NEXT: ret i1 %2
    a == b
}

// CHECK-LABEL: @array_eq_ref
#[no_mangle]
pub fn array_eq_ref(a: &[u16; 3], b: &[u16; 3]) -> bool {
    // CHECK: start:
    // CHECK: load i48, i48* %{{.+}}, align 2
    // CHECK: load i48, i48* %{{.+}}, align 2
    // CHECK: icmp eq i48
    // CHECK-NEXT: ret
    a == b
}

// CHECK-LABEL: @array_eq_value_still_passed_by_pointer
#[no_mangle]
pub fn array_eq_value_still_passed_by_pointer(a: [u16; 9], b: [u16; 9]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(18) %{{.+}}, i8* {{.*}} dereferenceable(18) %{{.+}}, i64 18)
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
    // CHECK-NEXT: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(i8* {{.*}} dereferenceable(2468) %{{.+}}, i8* {{.*}} dereferenceable(2468) %{{.+}}, i64 2468)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_zero_short(i48
#[no_mangle]
pub fn array_eq_zero_short(x: [u16; 3]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i48 %0, 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 3]
}

// CHECK-LABEL: @array_eq_zero_mid([8 x i16]*
#[no_mangle]
pub fn array_eq_zero_mid(x: [u16; 8]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: %[[LOAD:.+]] = load i128,
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i128 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 8]
}

// CHECK-LABEL: @array_eq_zero_long([1234 x i16]*
#[no_mangle]
pub fn array_eq_zero_long(x: [u16; 1234]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NOT: alloca
    // CHECK: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 1234]
}
