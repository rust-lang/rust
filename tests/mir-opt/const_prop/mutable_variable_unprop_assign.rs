// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes

// EMIT_MIR mutable_variable_unprop_assign.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug z => [[z:_.*]];
    // CHECK: [[a]] = foo()
    // CHECK: [[x]] = const (1_i32, 2_i32);
    // CHECK: ([[x]].1: i32) = copy [[a]];
    // CHECK: [[y]] = copy ([[x]].1: i32);
    // CHECK: [[z]] = copy ([[x]].0: i32);
    let a = foo();
    let mut x: (i32, i32) = (1, 2);
    x.1 = a;
    let y = x.1;
    let z = x.0;
}

#[inline(never)]
fn foo() -> i32 {
    unimplemented!()
}
