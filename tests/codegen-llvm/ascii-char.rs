//@ compile-flags: -C opt-level=1

#![crate_type = "lib"]
#![feature(ascii_char)]

use std::ascii::Char as AsciiChar;

// CHECK-LABEL: i8 @unwrap_digit_from_remainder(i32
#[no_mangle]
pub fn unwrap_digit_from_remainder(v: u32) -> AsciiChar {
    // CHECK-NOT: icmp
    // CHECK-NOT: panic

    // CHECK: %[[R:.+]] = urem i32 %v, 10
    // CHECK-NEXT: %[[T:.+]] = trunc{{( nuw)?( nsw)?}} i32 %[[R]] to i8
    // CHECK-NEXT: %[[D:.+]] = or{{( disjoint)?}} i8 %[[T]], 48
    // CHECK-NEXT: ret i8 %[[D]]

    // CHECK-NOT: icmp
    // CHECK-NOT: panic
    AsciiChar::digit((v % 10) as u8).unwrap()
}

// CHECK-LABEL: i8 @unwrap_from_masked(i8
#[no_mangle]
pub fn unwrap_from_masked(b: u8) -> AsciiChar {
    // CHECK-NOT: icmp
    // CHECK-NOT: panic

    // CHECK: %[[M:.+]] = and i8 %b, 127
    // CHECK-NEXT: ret i8 %[[M]]

    // CHECK-NOT: icmp
    // CHECK-NOT: panic
    AsciiChar::from_u8(b & 0x7f).unwrap()
}
