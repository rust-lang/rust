// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+InstSimplify
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR slice_len.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug local => [[local:_.*]];
    // CHECK: debug constant => [[constant:_.*]];

    // CHECK-NOT: {{_.*}} = Len(
    // CHECK-NOT: {{_.*}} = Lt(
    // CHECK-NOT: assert(move _
    // CHECK: {{_.*}} = const 3_usize;
    // CHECK: {{_.*}} = const true;
    // CHECK: assert(const true,

    // CHECK: [[local]] = (*{{_.*}})[1 of 2];
    let local = (&[1u32, 2, 3] as &[u32])[1];

    // CHECK-NOT: {{_.*}} = Len(
    // CHECK-NOT: {{_.*}} = Lt(
    // CHECK-NOT: assert(move _
    const SLICE: &[u32] = &[1, 2, 3];
    // CHECK: {{_.*}} = const 3_usize;
    // CHECK: {{_.*}} = const true;
    // CHECK: assert(const true,

    // CHECK-NOT: [[constant]] = (*{{_.*}})[_
    // CHECK: [[constant]] = (*{{_.*}})[1 of 2];
    let constant = SLICE[1];
}
