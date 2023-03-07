// compile-flags: -O -Z merge-functions=disabled
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
    // CHECK: load i48, {{i48\*|ptr}} %{{.+}}, align 2
    // CHECK: load i48, {{i48\*|ptr}} %{{.+}}, align 2
    // CHECK: icmp eq i48
    // CHECK-NEXT: ret
    a == b
}

// CHECK-LABEL: @array_eq_value_still_passed_by_pointer
#[no_mangle]
pub fn array_eq_value_still_passed_by_pointer(a: [u16; 9], b: [u16; 9]) -> bool {
    // CHECK-NEXT: start:
    // CHECK: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}({{i8\*|ptr}} {{.*}} dereferenceable(18) %{{.+}}, {{i8\*|ptr}} {{.*}} dereferenceable(18) %{{.+}}, i64 18)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_eq_long
#[no_mangle]
pub fn array_eq_long(a: &[u16; 1234], b: &[u16; 1234]) -> bool {
    // CHECK-NEXT: start:
    // CHECK: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}({{i8\*|ptr}} {{.*}} dereferenceable(2468) %{{.+}}, {{i8\*|ptr}} {{.*}} dereferenceable(2468) %{{.+}}, i64 2468)
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    a == b
}

// CHECK-LABEL: @array_char_eq
#[no_mangle]
pub fn array_char_eq(a: [char; 2], b: [char; 2]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i64 %0, %1
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

// CHECK-LABEL: @array_eq_none_short(i40
#[no_mangle]
pub fn array_eq_none_short(x: [Option<std::num::NonZeroU8>; 5]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i40 %0, 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [None; 5]
}

// CHECK-LABEL: @array_eq_zero_nested(
#[no_mangle]
pub fn array_eq_zero_nested(x: [[u8; 3]; 3]) -> bool {
    // CHECK: %[[VAL:.+]] = load i72
    // CHECK-SAME: align 1
    // CHECK: %[[EQ:.+]] = icmp eq i72 %[[VAL]], 0
    // CHECK: ret i1 %[[EQ]]
    x == [[0; 3]; 3]
}

// CHECK-LABEL: @array_eq_zero_mid(
#[no_mangle]
pub fn array_eq_zero_mid(x: [u16; 8]) -> bool {
    // CHECK-NEXT: start:
    // CHECK: %[[LOAD:.+]] = load i128,
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i128 %[[LOAD]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 8]
}

// CHECK-LABEL: @array_eq_zero_long(
#[no_mangle]
pub fn array_eq_zero_long(x: [u16; 1234]) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NOT: alloca
    // CHECK: %[[CMP:.+]] = tail call i32 @{{bcmp|memcmp}}(
    // CHECK-NEXT: %[[EQ:.+]] = icmp eq i32 %[[CMP]], 0
    // CHECK-NEXT: ret i1 %[[EQ]]
    x == [0; 1234]
}
