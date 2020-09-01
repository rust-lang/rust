// compile-flags: -O -Zmir-opt-level=3

// EMIT_MIR mult_by_zero.test.ConstProp.diff
fn test(x : i32) -> i32 {
  x * 0
}

fn main() {
    test(10);
}
