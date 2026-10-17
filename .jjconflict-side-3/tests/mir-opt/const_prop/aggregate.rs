// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -O

// EMIT_MIR aggregate.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK-NOT: = Add(
    // CHECK: [[x]] = const 1_u8;
    // CHECK-NOT: = Add(
    // CHECK: foo(const 1_u8)
    let x = (0, 1, 2).1 + 0;
    foo(x);
}

// Verify that we still propagate if part of the aggregate is not known.
// EMIT_MIR aggregate.foo.GVN.diff
fn foo(x: u8) {
    // CHECK-LABEL: fn foo(
    // CHECK: debug first => [[first:_.*]];
    // CHECK: debug second => [[second:_.*]];
    // CHECK-NOT: = Add(
    // CHECK: [[first]] = const 1_i32;
    // CHECK-NOT: = Add(
    // CHECK: [[second]] = const 3_i32;
    let first = (0, x).0 + 1;
    let second = (x, 1).1 + 2;
}
