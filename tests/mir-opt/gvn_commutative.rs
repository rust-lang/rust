//@ test-mir-pass: GVN

#![feature(core_intrinsics)]
#![crate_type = "lib"]

// EMIT_MIR gvn_commutative.add_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn add_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn add_commuted(
    // CHECK: [[LEFT:_.*]] = Add(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a + b, b + a)
}

// EMIT_MIR gvn_commutative.mul_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn mul_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn mul_commuted(
    // CHECK: [[LEFT:_.*]] = Mul(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a * b, b * a)
}

// EMIT_MIR gvn_commutative.and_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn and_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn and_commuted(
    // CHECK: [[LEFT:_.*]] = BitAnd(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a & b, b & a)
}

// EMIT_MIR gvn_commutative.or_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn or_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn or_commuted(
    // CHECK: [[LEFT:_.*]] = BitOr(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a | b, b | a)
}

// EMIT_MIR gvn_commutative.xor_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn xor_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn xor_commuted(
    // CHECK: [[LEFT:_.*]] = BitXor(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a ^ b, b ^ a)
}

// EMIT_MIR gvn_commutative.eq_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn eq_commuted(a: u32, b: u32) -> (bool, bool) {
    // CHECK-LABEL: fn eq_commuted(
    // CHECK: [[LEFT:_.*]] = Eq(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a == b, b == a)
}

// EMIT_MIR gvn_commutative.ne_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn ne_commuted(a: u32, b: u32) -> (bool, bool) {
    // CHECK-LABEL: fn ne_commuted(
    // CHECK: [[LEFT:_.*]] = Ne(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (a != b, b != a)
}

// EMIT_MIR gvn_commutative.add_unchecked_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn add_unchecked_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn add_unchecked_commuted(
    // CHECK: [[LEFT:_.*]] = AddUnchecked(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    unsafe { (core::intrinsics::unchecked_add(a, b), core::intrinsics::unchecked_add(b, a)) }
}

// EMIT_MIR gvn_commutative.mul_unchecked_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn mul_unchecked_commuted(a: u32, b: u32) -> (u32, u32) {
    // CHECK-LABEL: fn mul_unchecked_commuted(
    // CHECK: [[LEFT:_.*]] = MulUnchecked(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    unsafe { (core::intrinsics::unchecked_mul(a, b), core::intrinsics::unchecked_mul(b, a)) }
}

// EMIT_MIR gvn_commutative.add_overflow_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn add_overflow_commuted(a: u32, b: u32) -> ((u32, bool), (u32, bool)) {
    // CHECK-LABEL: fn add_overflow_commuted(
    // CHECK: [[LEFT:_.*]] = AddWithOverflow(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (core::intrinsics::add_with_overflow(a, b), core::intrinsics::add_with_overflow(b, a))
}

// EMIT_MIR gvn_commutative.mul_overflow_commuted.GVN.diff
#[unsafe(no_mangle)]
pub fn mul_overflow_commuted(a: u32, b: u32) -> ((u32, bool), (u32, bool)) {
    // CHECK-LABEL: fn mul_overflow_commuted(
    // CHECK: [[LEFT:_.*]] = MulWithOverflow(copy _1, copy _2);
    // CHECK: _0 = (copy [[LEFT]], copy [[LEFT]]);
    (core::intrinsics::mul_with_overflow(a, b), core::intrinsics::mul_with_overflow(b, a))
}
