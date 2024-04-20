// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN

// Due to a bug in propagating scalar pairs the assertion below used to fail. In the expected
// outputs below, after GVN this is how _2 would look like with the bug:
//
//     _2 = (const Scalar(0x00) : (), const 0u8);
//
// Which has the wrong type.

fn encode(this: ((), u8, u8)) {
    assert!(this.2 == 0);
}

// EMIT_MIR issue_66971.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: = encode(const ((), 0_u8, 0_u8))
    encode(((), 0, 0));
}
