//@ test-mir-pass: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR scalar_literal_propagation.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: = consume(const 1_u32)
    let x = 1;
    consume(x);
}

#[inline(never)]
fn consume(_: u32) {}
