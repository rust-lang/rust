// unit-test: SimplifyAutorefComparisons
// ignore-wasm32 compiled with panic=abort by default

#![crate_type = "lib"]

// EMIT_MIR simplify_cmp.ref_prim.SimplifyRefComparisons.diff
pub fn ref_prim(x: &i32, y: &i32) {
    let _a = x == y;
    let _b = x != y;
    let _c = x < y;
    let _d = x <= y;
    let _e = x > y;
    let _f = x >= y;
}

// EMIT_MIR simplify_cmp.multi_ref_prim.SimplifyRefComparisons.diff
pub fn multi_ref_prim(x: &&&i32, y: &&&i32) {
    let _a = x == y;
    let _b = x != y;
    let _c = x < y;
    let _d = x <= y;
    let _e = x > y;
    let _f = x >= y;
}

// EMIT_MIR simplify_cmp.ref_slice.SimplifyRefComparisons.diff
pub fn ref_slice(x: &str, y: &str) {
    let _a = x == y;
    let _b = x != y;
    let _c = x < y;
    let _d = x <= y;
    let _e = x > y;
    let _f = x >= y;
}

// EMIT_MIR simplify_cmp.ref_different_levels.SimplifyRefComparisons.diff
pub fn ref_different_levels(x: &&str, y: &String) {
    let _a = x == y;
    let _b = x != y;
    let _c = y == x;
    let _d = y != x;
}
