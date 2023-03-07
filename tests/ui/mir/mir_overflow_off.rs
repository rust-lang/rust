// run-pass
// compile-flags: -C overflow-checks=off

// Test that with MIR codegen, overflow checks can be
// turned off, even when they're from core::ops::*.

use std::ops::*;

fn main() {
    assert_eq!(i8::neg(-0x80), -0x80);

    assert_eq!(u8::add(0xff, 1), 0_u8);
    assert_eq!(u8::sub(0, 1), 0xff_u8);
    assert_eq!(u8::mul(0xff, 2), 0xfe_u8);
    assert_eq!(u8::shl(1, 9), 2_u8);
    assert_eq!(u8::shr(2, 9), 1_u8);
}
