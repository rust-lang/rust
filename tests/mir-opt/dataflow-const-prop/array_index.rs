// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DataflowConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR array_index.main.DataflowConstProp.diff

// CHECK-LABEL: fn main() -> () {
fn main() {
    // CHECK: let mut [[array_lit:_.*]]: [u32; 4];
    // CHECK:     debug x => [[x:_.*]];

    // CHECK:       [[array_lit]] = [const 0_u32, const 1_u32, const 2_u32, const 3_u32];
    // CHECK-NOT:   {{_.*}} = Len(
    // CHECK-NOT:   {{_.*}} = PtrMetadata(
    // CHECK-NOT:   {{_.*}} = Lt(
    // CHECK-NOT:   assert(move _
    // CHECK:       {{_.*}} = const 2_usize;
    // CHECK:       {{_.*}} = const true;
    // CHECK:       assert(const true
    // CHECK:       [[x]] = copy [[array_lit]][2 of 3];
    let x: u32 = [0, 1, 2, 3][2];
}
