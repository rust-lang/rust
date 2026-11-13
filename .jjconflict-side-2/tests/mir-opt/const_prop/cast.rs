//@ test-mir-pass: GVN
// EMIT_MIR cast.main.GVN.diff

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[x]] = const 42_u32;
    // CHECK: [[y]] = const 42_u8;
    let x = 42u8 as u32;
    let y = 42u32 as u8;
}
