// compile-flags: -Z mir-opt-level=2

// This used to ICE in const-prop

fn test(this: ((u8, u8),)) {
    assert!((this.0).0 == 1);
}

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    test(((1, 2),));
}
