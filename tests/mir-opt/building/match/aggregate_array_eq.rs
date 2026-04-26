// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=0

// Verify that matching against a constant array pattern produces a single
// `PartialEq::eq` call rather than element-by-element comparisons.
// In const contexts, the aggregate comparison must NOT be used because
// `PartialEq` is not const-stable.

#![crate_type = "lib"]

// EMIT_MIR aggregate_array_eq.array_match.built.after.mir
pub fn array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn array_match(
    // CHECK: <[u8; 4] as PartialEq>::eq
    // CHECK-NOT: switchInt(copy _1[
    matches!(x, [1, 2, 3, 4])
}

pub enum MyEnum {
    A,
    B,
    C,
    D,
}

// Regression test for https://github.com/rust-lang/rust/issues/103073.
// EMIT_MIR aggregate_array_eq.try_from_matched.built.after.mir
pub fn try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
    // CHECK-LABEL: fn try_from_matched(
    // CHECK: <[u8; 4] as PartialEq>::eq
    // CHECK-NOT: switchInt(copy (*_2)[
    match &value {
        b"ABCD" => Ok(MyEnum::A),
        b"EFGH" => Ok(MyEnum::B),
        b"IJKL" => Ok(MyEnum::C),
        b"MNOP" => Ok(MyEnum::D),
        _ => Err(()),
    }
}

// In a const fn, the aggregate comparison must not be used because
// `PartialEq::eq` cannot be called during const evaluation.
// EMIT_MIR aggregate_array_eq.const_array_match.built.after.mir
pub const fn const_array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn const_array_match(
    // CHECK-NOT: PartialEq
    // CHECK: switchInt
    matches!(x, [1, 2, 3, 4])
}

// EMIT_MIR aggregate_array_eq.const_try_from_matched.built.after.mir
pub const fn const_try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
    // CHECK-LABEL: fn const_try_from_matched(
    // CHECK-NOT: PartialEq
    // CHECK: switchInt
    match &value {
        b"ABCD" => Ok(MyEnum::A),
        b"EFGH" => Ok(MyEnum::B),
        b"IJKL" => Ok(MyEnum::C),
        b"MNOP" => Ok(MyEnum::D),
        _ => Err(()),
    }
}
