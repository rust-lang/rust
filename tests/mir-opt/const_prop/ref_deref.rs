// unit-test: ConstProp

// EMIT_MIR ref_deref.main.ConstProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: [[a]] = (*{{_.*}});
    let a = *(&4);
}
