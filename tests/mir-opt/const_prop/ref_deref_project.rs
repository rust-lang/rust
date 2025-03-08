// This does not currently propagate (#67862)
//@ test-mir-pass: GVN

// EMIT_MIR ref_deref_project.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: [[a]] = const 5_i32;
    let a = *(&(4, 5).1);
}
