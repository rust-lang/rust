// unit-test: GVN

// EMIT_MIR ref_deref.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => const 4_i32;
    let a = *(&4);
}
