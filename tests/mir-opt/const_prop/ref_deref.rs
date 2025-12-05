//@ test-mir-pass: GVN

// EMIT_MIR ref_deref.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: [[a]] = const 4_i32;
    let a = *(&4);
}
