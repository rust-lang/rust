// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR boolean_identities.test.DataflowConstProp.diff
pub fn test(x: bool, y: bool) -> bool {
    (y | true) & (x & false)
}

fn main() {
    test(true, false);
}
