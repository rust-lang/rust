// This test checks that the call to memchr is optimized away when searching in small slices.

// compile-flags: -O

#![crate_type = "lib"]

type T = u8;

// CHECK-LABEL: @foo1
#[no_mangle]
pub fn foo1(x: T, data: &[T; 1]) -> bool {
    // CHECK-NOT: memchr
    data.contains(&x)
}

// CHECK-LABEL: @foo2
#[no_mangle]
pub fn foo2(x: T, data: &[T; 2]) -> bool {
    // CHECK-NOT: memchr
    data.contains(&x)
}

// CHECK-LABEL: @foo3
#[no_mangle]
pub fn foo3(x: T, data: &[T; 3]) -> bool {
    // CHECK-NOT: memchr
    data.contains(&x)
}

// CHECK-LABEL: @foo4
#[no_mangle]
pub fn foo4(x: T, data: &[T; 4]) -> bool {
    // CHECK-NOT: memchr
    data.contains(&x)
}

// CHECK-LABEL: @foo16
#[no_mangle]
pub fn foo16(x: T, data: &[T; 16]) -> bool {
    // CHECK-NOT: memchr
    data.contains(&x)
}
