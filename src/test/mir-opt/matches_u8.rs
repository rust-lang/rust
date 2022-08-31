// unit-test: MatchBranchSimplification

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR matches_u8.exhaustive_match.MatchBranchSimplification.diff
// EMIT_MIR matches_u8.exhaustive_match_i8.MatchBranchSimplification.diff

pub enum E {
    A,
    B,
}

#[no_mangle]
pub fn exhaustive_match(e: E) -> u8 {
    match e {
        E::A => 0,
        E::B => 1,
    }
}

#[no_mangle]
pub fn exhaustive_match_i8(e: E) -> i8 {
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
