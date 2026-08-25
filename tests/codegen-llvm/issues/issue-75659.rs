// This test checks that the call to memchr/slice_contains is optimized away
// when searching in small slices.

//@ compile-flags: -Copt-level=3 -Zinline-mir=false
//@ only-x86_64

#![crate_type = "lib"]

// CHECK-LABEL: @foo1
#[no_mangle]
pub fn foo1(x: u8, data: &[u8; 1]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    data.contains(&x)
}

// CHECK-LABEL: @foo2
#[no_mangle]
pub fn foo2(x: u8, data: &[u8; 2]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    data.contains(&x)
}

// CHECK-LABEL: @foo3
#[no_mangle]
pub fn foo3(x: u8, data: &[u8; 3]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    data.contains(&x)
}

// CHECK-LABEL: @foo4
#[no_mangle]
pub fn foo4(x: u8, data: &[u8; 4]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    data.contains(&x)
}

// CHECK-LABEL: @foo8
#[no_mangle]
pub fn foo8(x: u8, data: &[u8; 8]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    data.contains(&x)
}

// CHECK-LABEL: @foo8_i8
#[no_mangle]
pub fn foo8_i8(x: i8, data: &[i8; 8]) -> bool {
    // CHECK-NOT: memchr
    // CHECK-NOT: slice_contains
    !data.contains(&x)
}

// Check that the general case isn't inlined
// CHECK-LABEL: @foo80
#[no_mangle]
pub fn foo80(x: u8, data: &[u8; 80]) -> bool {
    // CHECK: call core::slice::memchr
    data.contains(&x)
}
