// unit-test: DataflowConstProp
// compile-flags: -O -Zmir-opt-level=4

// EMIT_MIR boolean_identities.test.DataflowConstProp.diff
pub fn test(x: bool, y: bool) -> bool {
    (y | true) & (x & false)
}

fn main() {
    test(true, false);
}
