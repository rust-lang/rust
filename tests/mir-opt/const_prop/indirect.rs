// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -C overflow-checks=on

// EMIT_MIR indirect.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: [[x]] = const 3_u8;
    let x = (2u32 as u8) + 1;
}
