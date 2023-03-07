// unit-test: ConstProp
// compile-flags: -Z mir-opt-level=3

// Due to a bug in propagating scalar pairs the assertion below used to fail. In the expected
// outputs below, after ConstProp this is how _2 would look like with the bug:
//
//     _2 = (const Scalar(0x00) : (), const 0u8);
//
// Which has the wrong type.

fn encode(this: ((), u8, u8)) {
    assert!(this.2 == 0);
}

// EMIT_MIR issue_66971.main.ConstProp.diff
fn main() {
    encode(((), 0, 0));
}
