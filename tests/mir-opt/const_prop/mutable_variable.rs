// unit-test: ConstProp

// EMIT_MIR mutable_variable.main.ConstProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[x]] = const 42_i32;
    // CHECK: [[x]] = const 99_i32;
    // CHECK: [[y]] = const 99_i32;
    let mut x = 42;
    x = 99;
    let y = x;
}
