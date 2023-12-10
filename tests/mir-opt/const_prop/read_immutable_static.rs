// unit-test: ConstProp

static FOO: u8 = 2;

// EMIT_MIR read_immutable_static.main.ConstProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: [[x]] = const 4_u8;
    let x = FOO + FOO;
}
