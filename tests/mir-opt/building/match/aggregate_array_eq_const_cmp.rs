// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=0

// Verify that with `#![feature(const_cmp)]` and `#![feature(const_trait_impl)]`,
// const functions also use aggregate `PartialEq::eq` comparisons for constant
// array patterns, matching the behaviour of non-const functions.

#![crate_type = "lib"]
#![feature(const_cmp)]
#![feature(const_trait_impl)]

pub enum MyEnum {
    A,
    B,
    C,
    D,
}

// EMIT_MIR aggregate_array_eq_const_cmp.const_array_match.built.after.mir
pub const fn const_array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn const_array_match(
    // CHECK: <[u8; 4] as PartialEq>::eq
    // CHECK-NOT: switchInt(copy _1[
    matches!(x, [1, 2, 3, 4])
}

// EMIT_MIR aggregate_array_eq_const_cmp.const_try_from_matched.built.after.mir
pub const fn const_try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
    // CHECK-LABEL: fn const_try_from_matched(
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
