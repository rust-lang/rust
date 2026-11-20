//@ test-mir-pass: InstSimplify-after-simplifycfg
#![crate_type = "lib"]

// This is the easy case, and the most plausible to run into in real code.
// EMIT_MIR aggregate_array.literals.InstSimplify-after-simplifycfg.diff
pub fn literals() -> [u8; 5] {
    // CHECK-LABEL: fn literals(
    // CHECK: _0 = [const 0_u8; 5];
    [0, 0, 0, 0, 0]
}

// Check that hiding the const value behind a const item doesn't prevent the optimization
// EMIT_MIR aggregate_array.const_items.InstSimplify-after-simplifycfg.diff
pub fn const_items() -> [u8; 5] {
    const A: u8 = 0;
    const B: u8 = 0;
    const C: u8 = 0;
    const D: u8 = 0;
    const E: u8 = 0;

    // CHECK-LABEL: fn const_items(
    // CHECK: _0 = [const const_items::A; 5];
    [A, B, C, D, E]
}

// EMIT_MIR aggregate_array.strs.InstSimplify-after-simplifycfg.diff
pub fn strs() -> [&'static str; 5] {
    // CHECK-LABEL: fn strs(
    // CHECK: _0 = [const "a"; 5];
    ["a", "a", "a", "a", "a"]
}

// InstSimplify isn't able to see through the move operands, but GVN can.
// EMIT_MIR aggregate_array.local.InstSimplify-after-simplifycfg.diff
pub fn local() -> [u8; 5] {
    // CHECK-LABEL: fn local(
    // CHECK: _0 = [move _2, move _3, move _4, move _5, move _6];
    let val = 0;
    [val, val, val, val, val]
}

// All of these consts refer to the same value, but the addresses are all different.
// It would be wrong to apply the optimization here.
// EMIT_MIR aggregate_array.equal_referents.InstSimplify-after-simplifycfg.diff
pub fn equal_referents() -> [&'static u8; 5] {
    const DATA: &[u8] = &[0, 0, 0, 0, 0];
    const A: &u8 = &DATA[0];
    const B: &u8 = &DATA[1];
    const C: &u8 = &DATA[2];
    const D: &u8 = &DATA[3];
    const E: &u8 = &DATA[4];

    // CHECK-LABEL: fn equal_referents(
    // CHECK: _0 = [const equal_referents::A, const equal_referents::B, const equal_referents::C, const equal_referents::D, const equal_referents::E];
    [A, B, C, D, E]
}
