//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR tuple_literal_propagation.main.GVN.diff

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: = consume(const (1_u32, 2_u32))
    let x = (1, 2);
    consume(x);
}

#[inline(never)]
fn consume(_: (u32, u32)) {}
