//@ test-mir-pass: GVN

static FOO: u8 = 2;

// EMIT_MIR read_immutable_static.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => const 4_u8;
    let x = FOO + FOO;
}
