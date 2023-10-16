// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR mult_by_zero.test.DataflowConstProp.diff
fn test(x : i32) -> i32 {
  x * 0
}

fn main() {
    test(10);
}
