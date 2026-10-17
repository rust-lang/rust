// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN

// EMIT_MIR mutable_variable_aggregate_partial_read.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[x]] = foo()
    // CHECK: ([[x]].1: i32) = const 99_i32;
    // CHECK: ([[x]].0: i32) = const 42_i32;
    // CHECK: [[y]] = copy ([[x]].1: i32);
    let mut x: (i32, i32) = foo();
    x.1 = 99;
    x.0 = 42;
    let y = x.1;
}

#[inline(never)]
fn foo() -> (i32, i32) {
    unimplemented!()
}
