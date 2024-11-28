//@ test-mir-pass: GVN

static FOO: u8 = 2;

// EMIT_MIR read_immutable_static.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // Disabled due to <https://github.com/rust-lang/rust/issues/130853>
    // COM: CHECK: [[x]] = const 4_u8;
    let x = FOO + FOO;
}
