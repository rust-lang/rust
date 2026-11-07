// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zmir-opt-level=0

// Verify that matching against an array/slice pattern of a bytewise-comparable
// primitive type that was expanded from a constant (a named constant or a
// byte-string literal) produces a single call to the `raw_eq` or `compare_bytes`
// intrinsic rather than element-by-element comparisons. The call must be marked
// as non-unwinding.
//
// Hand-written patterns and patterns in const contexts are still compared
// element by element.

#![crate_type = "lib"]

// EMIT_MIR aggregate_array_eq.array_match.built.after.mir
pub fn array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn array_match(
    // CHECK-NOT: _1[
    // CHECK: raw_eq::<[u8; 4]>(
    // CHECK-SAME: unwind unreachable
    // CHECK-NOT: _1[
    const EXPECTED: [u8; 4] = [1, 2, 3, 4];
    matches!(x, EXPECTED)
}

// EMIT_MIR aggregate_array_eq.handwritten_array_match.built.after.mir
pub fn handwritten_array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn handwritten_array_match(
    // CHECK-NOT: raw_eq
    // CHECK: switchInt(copy _1[0 of 4])
    // CHECK: switchInt(copy _1[1 of 4])
    // CHECK: switchInt(copy _1[2 of 4])
    // CHECK: switchInt(copy _1[3 of 4])
    matches!(x, [1, 2, 3, 4])
}

// Below a length threshold, constants are still compared element by element.
// EMIT_MIR aggregate_array_eq.short_array_match.built.after.mir
pub fn short_array_match(x: [u8; 3]) -> bool {
    // CHECK-LABEL: fn short_array_match(
    // CHECK-NOT: raw_eq
    // CHECK: switchInt(copy _1[0 of 3])
    // CHECK: switchInt(copy _1[1 of 3])
    // CHECK: switchInt(copy _1[2 of 3])
    const EXPECTED: [u8; 3] = [1, 2, 3];
    matches!(x, EXPECTED)
}

#[derive(PartialEq, Eq)]
pub struct Element(u8);

// The aggregate comparison is limited to bytewise-comparable primitive element
// types, whose bytes are equal if and only if their values are.
// EMIT_MIR aggregate_array_eq.custom_element_array_match.built.after.mir
pub fn custom_element_array_match(x: [Element; 4]) -> bool {
    // CHECK-LABEL: fn custom_element_array_match(
    // CHECK-NOT: raw_eq
    // CHECK: switchInt(copy (_1[0 of 4].0: u8))
    // CHECK: switchInt(copy (_1[1 of 4].0: u8))
    // CHECK: switchInt(copy (_1[2 of 4].0: u8))
    // CHECK: switchInt(copy (_1[3 of 4].0: u8))
    const EXPECTED: [Element; 4] = [Element(1), Element(2), Element(3), Element(4)];
    matches!(x, EXPECTED)
}

// Slice patterns of the same length share a single length check.
// EMIT_MIR aggregate_array_eq.slice_match.built.after.mir
pub fn slice_match(x: &[u8]) -> u32 {
    // CHECK-LABEL: fn slice_match(
    // CHECK: PtrMetadata(copy _1);
    // CHECK-NOT: PtrMetadata
    // CHECK-NOT: (*_1)[
    // CHECK: compare_bytes(
    // CHECK-SAME: unwind unreachable
    // CHECK-NOT: PtrMetadata
    // CHECK-NOT: (*_1)[
    // CHECK: compare_bytes(
    // CHECK-SAME: unwind unreachable
    // CHECK-NOT: PtrMetadata
    // CHECK-NOT: (*_1)[
    match x {
        b"ABCD" => 1,
        b"EFGH" => 2,
        _ => 0,
    }
}

// EMIT_MIR aggregate_array_eq.short_slice_match.built.after.mir
pub fn short_slice_match(x: &[u8]) -> bool {
    // CHECK-LABEL: fn short_slice_match(
    // CHECK-NOT: compare_bytes
    // CHECK: PtrMetadata(copy _1);
    // CHECK: switchInt(copy (*_1)[0 of 3])
    // CHECK: switchInt(copy (*_1)[1 of 3])
    // CHECK: switchInt(copy (*_1)[2 of 3])
    matches!(x, b"ABC")
}

// `compare_bytes` is given the length in bytes, not in elements.
// EMIT_MIR aggregate_array_eq.u16_slice_match.built.after.mir
pub fn u16_slice_match(x: &[u16]) -> bool {
    // CHECK-LABEL: fn u16_slice_match(
    // CHECK: [[BYTES:_[0-9]+]] = const 8_usize;
    // CHECK: compare_bytes({{.*}}, move [[BYTES]])
    const EXPECTED: &[u16] = &[1, 2, 3, 4];
    matches!(x, EXPECTED)
}

// EMIT_MIR aggregate_array_eq.char_slice_match.built.after.mir
pub fn char_slice_match(x: &[char]) -> bool {
    // CHECK-LABEL: fn char_slice_match(
    // CHECK: [[BYTES:_[0-9]+]] = const 16_usize;
    // CHECK: compare_bytes({{.*}}, move [[BYTES]])
    const EXPECTED: &[char] = &['a', 'b', 'c', 'd'];
    matches!(x, EXPECTED)
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
    // CHECK: [[SCRUTINEE:_[0-9]+]] = &_1;
    // CHECK-NOT: (*[[SCRUTINEE]])[
    // CHECK: raw_eq::<[u8; 4]>(
    // CHECK-NOT: (*[[SCRUTINEE]])[
    match &value {
        b"ABCD" => Ok(MyEnum::A),
        b"EFGH" => Ok(MyEnum::B),
        b"IJKL" => Ok(MyEnum::C),
        b"MNOP" => Ok(MyEnum::D),
        _ => Err(()),
    }
}

// In a const fn, the aggregate comparison must not be used because the
// `compare_bytes` intrinsic is not const-stable.
// EMIT_MIR aggregate_array_eq.const_array_match.built.after.mir
pub const fn const_array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn const_array_match(
    // CHECK-NOT: raw_eq
    // CHECK: switchInt(copy _1[0 of 4])
    // CHECK: switchInt(copy _1[1 of 4])
    // CHECK: switchInt(copy _1[2 of 4])
    // CHECK: switchInt(copy _1[3 of 4])
    const EXPECTED: [u8; 4] = [1, 2, 3, 4];
    matches!(x, EXPECTED)
}

// EMIT_MIR aggregate_array_eq.const_try_from_matched.built.after.mir
pub const fn const_try_from_matched(value: [u8; 4]) -> Result<MyEnum, ()> {
    // CHECK-LABEL: fn const_try_from_matched(
    // CHECK-NOT: raw_eq
    // CHECK: [[SCRUTINEE:_[0-9]+]] = &_1;
    // CHECK: switchInt(copy (*[[SCRUTINEE]])[0 of 4])
    // CHECK: switchInt(copy (*[[SCRUTINEE]])[1 of 4])
    // CHECK: switchInt(copy (*[[SCRUTINEE]])[2 of 4])
    // CHECK: switchInt(copy (*[[SCRUTINEE]])[3 of 4])
    match &value {
        b"ABCD" => Ok(MyEnum::A),
        b"EFGH" => Ok(MyEnum::B),
        b"IJKL" => Ok(MyEnum::C),
        b"MNOP" => Ok(MyEnum::D),
        _ => Err(()),
    }
}
