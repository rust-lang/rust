//@ test-mir-pass: MatchBranchSimplification

// EMIT_MIR matches_u8.exhaustive_match.MatchBranchSimplification.diff
// EMIT_MIR matches_u8.exhaustive_match_i8.MatchBranchSimplification.diff

pub enum E {
    A,
    B,
}

#[no_mangle]
pub fn exhaustive_match(e: E) -> u8 {
    // CHECK-LABEL: fn exhaustive_match(
    // CHECK: discriminant(
    // CHECK-NOT: switchInt
    // CHECK: return
    match e {
        E::A => 0,
        E::B => 1,
    }
}

#[no_mangle]
pub fn exhaustive_match_i8(e: E) -> i8 {
    // CHECK-LABEL: fn exhaustive_match_i8(
    // CHECK: discriminant(
    // CHECK-NOT: switchInt
    // CHECK: return
    match e {
        E::A => 0,
        E::B => 1,
    }
}

fn main() {
    assert_eq!(exhaustive_match(E::A), 0);
    assert_eq!(exhaustive_match(E::B), 1);

    assert_eq!(exhaustive_match_i8(E::A), 0);
    assert_eq!(exhaustive_match_i8(E::B), 1);
}
