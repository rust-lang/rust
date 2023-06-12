// compile-flags: -O -Zmir-opt-level=3

// EMIT_MIR ref_int_cmp.opt1.RefCmpSimplify.diff
pub fn opt1(x: &u8, y: &u8) -> bool {
  x < y
}

fn main() {
  opt1(&1, &2);
}
