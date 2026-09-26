//@ test-mir-pass: GVN,+LowerIntrinsics
//@ compile-flags: -C overflow-checks=off

#![feature(core_intrinsics)]
#![crate_type = "lib"]

unsafe extern "Rust" {
    safe fn consume_pair(x: u32, y: u32) -> u32;
    safe fn consume_bool_pair(x: bool, y: bool) -> u32;
    safe fn consume_overflow_pair(x: (u32, bool), y: (u32, bool)) -> u32;
}

#[unsafe(no_mangle)]
pub fn add_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn add_commuted(
    // CHECK: = Add(copy _1, copy _2);
    // CHECK-NOT: = Add(
    // CHECK: consume_pair(
    consume_pair(a + b, b + a)
}

#[unsafe(no_mangle)]
pub fn mul_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn mul_commuted(
    // CHECK: = Mul(copy _1, copy _2);
    // CHECK-NOT: = Mul(
    // CHECK: consume_pair(
    consume_pair(a * b, b * a)
}

#[unsafe(no_mangle)]
pub fn and_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn and_commuted(
    // CHECK: = BitAnd(copy _1, copy _2);
    // CHECK-NOT: = BitAnd(
    // CHECK: consume_pair(
    consume_pair(a & b, b & a)
}

#[unsafe(no_mangle)]
pub fn or_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn or_commuted(
    // CHECK: = BitOr(copy _1, copy _2);
    // CHECK-NOT: = BitOr(
    // CHECK: consume_pair(
    consume_pair(a | b, b | a)
}

#[unsafe(no_mangle)]
pub fn xor_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn xor_commuted(
    // CHECK: = BitXor(copy _1, copy _2);
    // CHECK-NOT: = BitXor(
    // CHECK: consume_pair(
    consume_pair(a ^ b, b ^ a)
}

#[unsafe(no_mangle)]
pub fn eq_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn eq_commuted(
    // CHECK: = Eq(copy _1, copy _2);
    // CHECK-NOT: = Eq(
    // CHECK: consume_bool_pair(
    consume_bool_pair(a == b, b == a)
}

#[unsafe(no_mangle)]
pub fn ne_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn ne_commuted(
    // CHECK: = Ne(copy _1, copy _2);
    // CHECK-NOT: = Ne(
    // CHECK: consume_bool_pair(
    consume_bool_pair(a != b, b != a)
}

#[unsafe(no_mangle)]
pub fn add_unchecked_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn add_unchecked_commuted(
    // CHECK: = AddUnchecked(copy _1, copy _2);
    // CHECK-NOT: = AddUnchecked(
    // CHECK: consume_pair(
    unsafe {
        consume_pair(core::intrinsics::unchecked_add(a, b), core::intrinsics::unchecked_add(b, a))
    }
}

#[unsafe(no_mangle)]
pub fn mul_unchecked_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn mul_unchecked_commuted(
    // CHECK: = MulUnchecked(copy _1, copy _2);
    // CHECK-NOT: = MulUnchecked(
    // CHECK: consume_pair(
    unsafe {
        consume_pair(core::intrinsics::unchecked_mul(a, b), core::intrinsics::unchecked_mul(b, a))
    }
}

#[unsafe(no_mangle)]
pub fn add_overflow_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn add_overflow_commuted(
    // CHECK: = AddWithOverflow(copy _1, copy _2);
    // CHECK-NOT: = AddWithOverflow(
    // CHECK: consume_overflow_pair(
    consume_overflow_pair(
        core::intrinsics::add_with_overflow(a, b),
        core::intrinsics::add_with_overflow(b, a),
    )
}

#[unsafe(no_mangle)]
pub fn mul_overflow_commuted(a: u32, b: u32) -> u32 {
    // CHECK-LABEL: fn mul_overflow_commuted(
    // CHECK: = MulWithOverflow(copy _1, copy _2);
    // CHECK-NOT: = MulWithOverflow(
    // CHECK: consume_overflow_pair(
    consume_overflow_pair(
        core::intrinsics::mul_with_overflow(a, b),
        core::intrinsics::mul_with_overflow(b, a),
    )
}
