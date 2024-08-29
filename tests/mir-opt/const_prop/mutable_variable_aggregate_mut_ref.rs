//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes

// EMIT_MIR mutable_variable_aggregate_mut_ref.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug z => [[z:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[x]] = const (42_i32, 43_i32);
    // CHECK: [[z]] = &mut [[x]];
    // CHECK: ((*[[z]]).1: i32) = const 99_i32;
    // CHECK: [[y]] = copy [[x]];
    let mut x = (42, 43);
    let z = &mut x;
    z.1 = 99;
    let y = x;
}
