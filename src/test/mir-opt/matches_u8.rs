// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR matches_u8.exhaustive_match.MatchBranchSimplification.diff

pub enum E {
    A,
    B,
}

// This only breaks on u8's, but probably still have to test i8.
#[no_mangle]
pub fn exhaustive_match(e: E) -> u8 {
    match e {
        E::A => 0,
        E::B => 1,
    }
}

fn main() {
  assert_eq!(exhaustive_match(E::A), 0);
  assert_eq!(exhaustive_match(E::B), 1);
}
