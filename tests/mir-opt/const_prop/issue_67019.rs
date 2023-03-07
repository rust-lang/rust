// unit-test: ConstProp
// compile-flags: -Z mir-opt-level=3

// This used to ICE in const-prop

fn test(this: ((u8, u8),)) {
    assert!((this.0).0 == 1);
}

// EMIT_MIR issue_67019.main.ConstProp.diff
fn main() {
    test(((1, 2),));
}
