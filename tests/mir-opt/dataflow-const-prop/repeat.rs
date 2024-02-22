//@ unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR repeat.main.DataflowConstProp.diff
// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug x => [[x:_.*]];

    // CHECK: [[array_lit:_.*]] = [const 42_u32; 8];
    // CHECK-NOT: {{_.*}} = Len(
    // CHECK-NOT: {{_.*}} = Lt(
    // CHECK: {{_.*}} = const 8_usize;
    // CHECK: {{_.*}} = const true;
    // CHECK: assert(const true

    // CHECK-NOT: [[t:_.*]] = [[array_lit]][_
    // CHECK: [[t:_.*]] = [[array_lit]][2 of 3];
    // CHECK: [[x]] = Add(move [[t]], const 0_u32);
    let x: u32 = [42; 8][2] + 0;
}
